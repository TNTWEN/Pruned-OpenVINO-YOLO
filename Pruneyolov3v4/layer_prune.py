from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.utils import *
from utils.prune_utils import *
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-hand.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/oxfordhand.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--shortcuts', type=int, default=8, help='how many shortcut layers will be pruned,\
        pruning one shortcut will also prune two CBL,yolov3 has 23 shortcuts')
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
        load_darknet_weights(model, opt.weights)
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

    with torch.no_grad():
        print("\nlet's test the original model first:")
        origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)

    # shortcut_idx是shortcut前一个卷积模块的索引
    CBL_idx, Conv_idx, shortcut_idx = parse_module_defs4(model.module_defs)
    print('all shortcut_idx:', [i + 1 for i in shortcut_idx])


    bn_weights = gather_bn_weights(model.module_list, shortcut_idx)

    sorted_bn = torch.sort(bn_weights)[0]


    # highest_thre = torch.zeros(len(shortcut_idx))
    # for i, idx in enumerate(shortcut_idx):
    #     highest_thre[i] = model.module_list[idx][1].weight.data.abs().max().clone()
    # _, sorted_index_thre = torch.sort(highest_thre)
    
    #这里更改了选层策略，由最大值排序改为均值排序，均值一般表现要稍好，但不是绝对，可以自己切换尝试；前面注释的四行为原策略。
    bn_mean = torch.zeros(len(shortcut_idx)) # bn_mean代表不同shortcut前一层的卷积模块所有gamma系数的均值
    for i, idx in enumerate(shortcut_idx):
        bn_mean[i] = model.module_list[idx][1].weight.data.abs().mean().clone()
    _, sorted_index_thre = torch.sort(bn_mean)
    

    # prune_shortcuts表示需要被剪枝的shortcut（实际上剪枝的是shortcut和它前面的两个卷积模块，bottleneck）
    prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:opt.shortcuts]]] 
    prune_shortcuts = [int(x) for x in prune_shortcuts] 

    index_all = list(range(len(model.module_defs))) 
    index_prune = [] # 需要被剪枝的模块构成的索引列表
    for idx in prune_shortcuts:
        index_prune.extend([idx - 1, idx, idx + 1])
    index_remain = [idx for idx in index_all if idx not in index_prune] # 剩余的没被剪枝的模块构成的索引列表

    print('These shortcut layers and corresponding CBL will be pruned :', index_prune)





    def prune_and_eval(model, prune_shortcuts=[]):
        model_copy = deepcopy(model)
        for idx in prune_shortcuts:
            for i in [idx, idx-1]: # 只需要把shortcut的前两个卷积模块中BN的gamma系数置为0即可达到剪枝该shortcut的效果
                bn_module = model_copy.module_list[i][1]

                mask = torch.zeros(bn_module.weight.data.shape[0]).cuda()
                bn_module.weight.data.mul_(mask)
         

        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]

        print(f'simply mask the BN Gama of to_be_pruned CBL as zero, now the mAP is {mAP:.4f}')


    prune_and_eval(model, prune_shortcuts)

    #%%
    def obtain_filters_mask(model, CBL_idx, prune_shortcuts):

        filters_mask = []
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            mask = np.ones(bn_module.weight.data.shape[0], dtype='float32')
            filters_mask.append(mask.copy())
        CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
        for idx in prune_shortcuts:
            for i in [idx, idx - 1]:
                bn_module = model.module_list[i][1]
                mask = np.zeros(bn_module.weight.data.shape[0], dtype='float32')
                CBLidx2mask[i] = mask.copy()
        return CBLidx2mask

    # CBLidx2mask中（key,value），key为包含BN层的卷积模块的索引，value为该卷积模块的剪枝掩码数组，其中数组中的元素要么全为1（保留），要么全为0（剪枝） 
    CBLidx2mask = obtain_filters_mask(model, CBL_idx, prune_shortcuts) 

    pruned_model = prune_model_keep_size2(model, CBL_idx, CBL_idx, CBLidx2mask)

    with torch.no_grad():
        mAP = eval_model(pruned_model)[0][2]
    print("after transfering the offset of pruned CBL's activation, map is {}".format(mAP))

    compact_module_defs = deepcopy(model.module_defs)


    for j, module_def in enumerate(compact_module_defs):    
        if module_def['type'] == 'route':
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            """
            from_layers = [int(s) for s in module_def['layers'].split(',')]
            if len(from_layers) == 1 and from_layers[0] > 0: # route层为单输入时
                count = 0
                for i in index_prune: # 统计单输入的route层前面有多少个模块被剪枝了
                    if i <= from_layers[0]:
                        count += 1
                from_layers[0] = from_layers[0] - count 
                from_layers = str(from_layers[0])
                module_def['layers'] = from_layers # 修改route层的‘layers’配置参数

            elif len(from_layers) == 2: # route层为双输入时
                count = 0
                if from_layers[1] > 0:
                    for i in index_prune:
                        if i <= from_layers[1]: # 被剪枝的模块在route层的前面
                            count += 1
                    from_layers[1] = from_layers[1] - count
                else:
                    for i in index_prune:
                        if i > j + from_layers[1] and i < j: # 被剪枝的模块在route层的后面
                            count += 1
                    from_layers[1] = from_layers[1] + count

                from_layers = ', '.join([str(s) for s in from_layers])
                module_def['layers'] = from_layers
            """
            from_layers = module_def['layers']

            if len(from_layers) == 1 and from_layers[0] > 0: # route层为单输入时
                count = 0
                for i in index_prune: # 统计单输入的route层前面有多少个模块被剪枝了
                    if i <= from_layers[0]:
                        count += 1
                from_layers[0] = from_layers[0] - count 
                module_def['layers'] = from_layers # 修改route层的‘layers’配置参数

            elif len(from_layers) == 2: # route层为双输入时
                count = 0
                if from_layers[1] > 0:
                    for i in index_prune:
                        if i <= from_layers[1]: # 被剪枝的模块在route层的前面
                            count += 1
                    from_layers[1] = from_layers[1] - count
                else:
                    for i in index_prune:
                        if i > j + from_layers[1] and i < j: # 被剪枝的模块在route层的后面
                            count += 1
                    from_layers[1] = from_layers[1] + count

                module_def['layers'] = from_layers
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
    
    compact_module_defs = [compact_module_defs[i] for i in index_remain] # 通过剩余模块的索引构建剪枝完成后的紧凑模型
    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size), verbose=False).to(device)

    for i, index in enumerate(index_remain):
        #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
        # compact_model.module_list[i] = pruned_model.module_list[index] # 将原模型的参数拷贝到剪枝完成后的紧凑模型中
        compact_model.module_list[i] = deepcopy(pruned_model.module_list[index]) # 将原模型的参数拷贝到剪枝完成后的紧凑模型中
        
        if compact_module_defs[i]['type'] == 'route':
            compact_model.module_list[i].multiple = len(compact_module_defs[i]['layers']) > 1
            compact_model.module_list[i].layers = compact_module_defs[i]['layers']
        #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#

    compact_nparameters = obtain_num_parameters(compact_model)

    # init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)
    
    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output
    
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)
    

    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
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
    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{opt.shortcuts}_shortcut_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.weights.replace('/', f'/prune_{opt.shortcuts}_shortcut_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')

    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')



