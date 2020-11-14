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
    parser.add_argument('--weights', type=str, default='weights/yolov3', help='sparse model weights')
    parser.add_argument('--global_percent', type=float, default=0.8, help='global channel prune percent')
    parser.add_argument('--layer_keep', type=float, default=0.01, help='channel keep percent per layer')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)
    #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
    model.hyperparams["cfg_path"]=opt.cfg
    #+++++++++++++++++++++++++ insert end++++++++++++++++++++++# 

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
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
    
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)


    """
    CBL_idx = [] # 包含BN层的卷积模块的索引列表（CBL: Conv-Bn-Leaky_relu ）
    Conv_idx = [] # 不包含BN层的卷积模块的索引列表

    ignore_idx = set() # 不能够被剪枝的卷积模块的索引列表（spp前一个CBL不剪,上采样层前的卷积模块，...）
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx] # 能够被剪枝的模块（包含BN层但是不在ignore_idx里的卷积模块）
    """
    CBL_idx, Conv_idx, prune_idx, _, _= parse_module_defs2(model.module_defs)



    bn_weights = gather_bn_weights(model.module_list, prune_idx)#一维tensor，大小是所有可prune的BN层的size和.将所有的weights（gamma）都放在里面

    sorted_bn = torch.sort(bn_weights)[0]
    sorted_bn, sorted_index = torch.sort(bn_weights)#从小到大排序，  原来对应位置序号
    thresh_index = int(len(bn_weights) * opt.global_percent)
    thresh = sorted_bn[thresh_index].cuda()

    print(f'Global Threshold should be less than {thresh:.4f}.')




    #%%
    def obtain_filters_mask(model, thre, CBL_idx, prune_idx):

        pruned = 0
        total = 0
        num_filters = []#每个CBL剩下的通道数量int值
        filters_mask = []#每个CBL的mask
        for idx in CBL_idx: # 遍历所有包含BN层的卷积模块
            bn_module = model.module_list[idx][1] # BN层
            if idx in prune_idx:#如果是可prune的

                weight_copy = bn_module.weight.data.abs().clone() # BN层的gamma系数
                
                channels = weight_copy.shape[0] # 当前卷积模块的输出通道数目
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 1 # 当前卷积模块最少要保留的输出通道数目
                mask = weight_copy.gt(thresh).float() # 当前卷积模块的剪枝掩码数组（数组元素为1代表保留，为0代表剪枝） .gt大于则为1，不大于则为0  .float() true->1.0 false=0.0
                
                if int(torch.sum(mask)) < min_channel_num: # 当剪枝后输出通道数目小于当前卷积模块最少要保留的输出通道数目
                    _, sorted_index_weights = torch.sort(weight_copy,descending=True)
                    mask[sorted_index_weights[:min_channel_num]]=1.  #mask更新为min_channel_num个
                remain = int(mask.sum()) # 剪枝后剩余的输出通道数目
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                        f'remaining channel: {remain:>4d}')
            else:#不可prune的全部保留
                mask = torch.ones(bn_module.weight.data.shape)#全1保留
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.clone())

        prune_ratio = pruned / total
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

        return num_filters, filters_mask

    num_filters, filters_mask = obtain_filters_mask(model, thresh, CBL_idx, prune_idx)#每个CBL剩下的通道数量int值  每个CBL的mask  ignore的就是通道全部保留，mask全是1
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)} #（key,value）:key为被剪枝的卷积模块的索引，value为该卷积模块的剪枝掩码
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)} #（key,value）:key为被剪枝的卷积模块的索引，value为该卷积模块的剩余的输出通道数目

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i['is_access'] = False


    print('merge the mask of layers connected to shortcut!')
    """
    重点：
        * 剪枝掩码对1求并集，即但凡有一个卷积模块中的某个位置的输出通道不被剪枝，则所有卷积模块在该位置的输出通道都不被剪枝
        * 即剪枝掩码对0求交集,即只有所有卷积模块在某个位置的输出通道被剪枝，则所有卷积模块在该位置的输出通道才能被剪枝
    """
    merge_mask(model, CBLidx2mask, CBLidx2filters) 



    def prune_and_eval(model, CBL_idx, CBLidx2mask):
        model_copy = deepcopy(model)

        for idx in CBL_idx:
            bn_module = model_copy.module_list[idx][1]
            mask = CBLidx2mask[idx].cuda()
            bn_module.weight.data.mul_(mask)#mul 对应位相乘

        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]

        print(f'mask the gamma as zero, mAP of the model is {mAP:.4f}')


    prune_and_eval(model, CBL_idx, CBLidx2mask)


    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()



    pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
    print("\nnow prune the model but keep size,(actually add offset of BN beta to following layers), let's see how the mAP goes")

    with torch.no_grad():
        eval_model(pruned_model)

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i.pop('is_access')

    compact_module_defs = deepcopy(model.module_defs)
    for idx in CBL_idx:
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(CBLidx2filters[idx])


    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)#list加法 合并
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

    print('testing inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)


    print('testing the final model...')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)


    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)



    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{opt.global_percent}_keep_{opt.layer_keep}_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.weights.replace('/', f'/prune_{opt.global_percent}_keep_{opt.layer_keep}_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')

