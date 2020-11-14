from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.6, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device) # 根据配置文件生成模型
    #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
    model.hyperparams["cfg_path"]=opt.cfg
    #+++++++++++++++++++++++++ insert end++++++++++++++++++++++# 

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model']) # 加载pytorch默认格式的模型参数
    else:
        _ = load_darknet_weights(model, opt.weights) # 加载darknet格式的模型参数
    print('\nloaded weights from ',opt.weights)

    #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
    """
    eval_model = lambda model:test(model=model,cfg=opt.cfg, data=opt.data, batch_size=16, img_size=img_size) # 用于模型测试的lambda函数
    """
    eval_model = lambda model:test(opt.cfg, opt.data, 
                weights=opt.weights, 
                batch_size=16,
                imgsz=img_size,
                iou_thres=0.5,
                conf_thres=0.001,
                save_json=False,
                model=model)
    #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()]) # 用于获取模型参数数量的lambda函数

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model) # 对未剪枝的模型进行测试
    origin_nparameters = obtain_num_parameters(model) # 获取未剪枝模型的参数数量

    """
    CBL_idx = [] # 包含BN层的卷积模块的索引列表（CBL: Conv-Bn-Leaky_relu）
    Conv_idx = [] # 不包含BN层的卷积模块的索引列表

    ignore_idx = set() # 不能够被剪枝的卷积模块的索引列表（spp前一个CBL不剪,上采样层前的卷积模块，...）
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx] # 能够被剪枝的模块（包含BN层但是不在ignore_idx里的卷积模块）

    shortcut_idx=dict() # (key,value),其中key，value为一对生成shortcut模块的两个输入特征图的卷积模块，key为直连接，value为跨层连接
    shortcut_all=set() # 生成shortcut模块的输入特征图的卷积模块的索引集合，每个shortcut模块对应两个这样的卷积模块
    """
    CBL_idx, Conv_idx, prune_idx,shortcut_idx,shortcut_all= parse_module_defs2(model.module_defs) # 对模型进行解析

    # 从能够被剪枝的模块中去掉生成shortcut直连输入特征图的卷积模块，注意这里的sort_prune_idx依然包含生成shortcut跨层输入特征图的卷积模块
    sort_prune_idx=[idx for idx in prune_idx if idx not in shortcut_idx] 

    #将所有要剪枝的BN层的gamma系数，拷贝到bn_weights列表
    bn_weights = gather_bn_weights(model.module_list, sort_prune_idx)

    #torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
    sorted_bn = torch.sort(bn_weights)[0]


    #避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for idx in sort_prune_idx:
        #.item()可以得到张量里的元素值
        highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
    highest_thre = min(highest_thre)

    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)

    print(f'Suggested Threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f},but you can set higher.')


    def prune_and_eval(model, sorted_bn, percent=.0):
        model_copy = deepcopy(model)
        thre_index = int(len(sorted_bn) * percent)
        #获得gamma系数的阈值，小于该值的gamma系数对应的通道，全部裁剪掉
        thre1 = sorted_bn[thre_index]

        print(f'Channels with Gamma value less than {thre1:.6f} are pruned!')

        remain_num = 0
        idx_new=dict() # (key,value): key为模块idx，value为模块的剪枝掩码数组（元素值为1代表该滤波器保留，为0代表剪枝）
        for idx in prune_idx: # 这里的prune_idx是包含生成shortcut输入的卷积模块
            
            if idx not in shortcut_idx: # 当idx不是生成shortcut输入的卷积模块
                
                bn_module = model_copy.module_list[idx][1]

                mask = obtain_bn_mask(bn_module, thre1) # 根据thre1判断当前模块中的哪些输出通道需要被剪枝（通过设置掩码为0来表示该滤波器被剪枝了）
                #记录剪枝后，每一层卷积层对应的mask
                # idx_new[idx]=mask.cpu().numpy()
                idx_new[idx]=mask # 记录当前模块的剪枝掩码

                remain_num += int(mask.sum()) # 未被剪枝的输出通道数目
                bn_module.weight.data.mul_(mask) # 将被剪枝的输出通道对应的gamma系数置为0
                #bn_module.bias.data.mul_(mask*0.0001)

            else: 
                # 当idx是生成shortcut的直连输入特征图的卷积模块（注意，这里就是与prune.py不同的关键了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!）
                # 即当shortcut跨层连接中的某个输出通道被剪枝了，那么对应的shortcut直连接中对应的输出通道也要被剪枝（前面剪完，后面对应也要剪）
                
                bn_module = model_copy.module_list[idx][1]
               
                # shortcut_idx[idx]为生成shortcut跨层输入特征图的卷积模块的索引
                # idx_new[shortcut_idx[idx]]代表生成shortcut跨层输入特征图的卷积模块对应的剪枝掩码数组
                mask=idx_new[shortcut_idx[idx]] 
                idx_new[idx]=mask # 记录当前模块的剪枝掩码
     
                remain_num += int(mask.sum()) # 未被剪枝的输出通道数目 （理论上这里的这行代码应该要删除，因为shortcut直连接中对应的输出通道不被包含于sorted_bn）
                bn_module.weight.data.mul_(mask) # 将被剪枝的输出通道对应的gamma系数置为0
                
            #print(int(mask.sum()))

        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]

        print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
        print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')
        print(f'mAP of the pruned model is {mAP:.4f}')

        return thre1

    percent = opt.percent
    threshold = prune_and_eval(model, sorted_bn, percent)



    #****************************************************************
    #虽然上面已经能看到剪枝后的效果，但是没有生成剪枝后的模型结构，因此下面的代码是为了生成新的模型结构并拷贝旧模型参数到新模型



    #%%
    def obtain_filters_mask(model, thre, CBL_idx, prune_idx):

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        idx_new=dict()
        #CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:
                if idx not in shortcut_idx:

                    mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
                    idx_new[idx]=mask
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain

                    # if remain == 0:
                    #     print("Channels would be all pruned!")
                    #     raise Exception

                    # print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                    #     f'remaining channel: {remain:>4d}')
                else:
                    mask=idx_new[shortcut_idx[idx]]
                    idx_new[idx]=mask
                    remain= int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                    
                if remain == 0:
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = bn_module.weight.data.abs().max()
                    mask = obtain_bn_mask(bn_module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                        f'remaining channel: {remain:>4d}')
            else:
                mask = np.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())

        #因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数
        prune_ratio = pruned / total
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

        return num_filters, filters_mask

    num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)


    #CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

    pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
    print("\nnow prune the model but keep size,(actually add offset of BN beta to next layer), let's see how the mAP goes")

    with torch.no_grad():
        eval_model(pruned_model)

    #获得原始模型的module_defs，并修改该defs中的卷积核数量
    compact_module_defs = deepcopy(model.module_defs)
    for idx, num in zip(CBL_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)


    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)


    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):

        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output

    print('testing Inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)


    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing final model')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)


    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)


    # 生成剪枝后的cfg文件并保存模型
    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{percent}_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.weights.replace('/', f'/prune_{percent}_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')

