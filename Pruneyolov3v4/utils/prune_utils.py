import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            # if module_def['batch_normalize'] == '1':
            if module_def['batch_normalize'] == 1:
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                #spp前一个CBL不剪 区分tiny
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'route' and 'groups' in module_defs[i+1]:
                ignore_idx.add(i)

        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)

            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            # identity_idx = (i + int(module_def['from'])) # shortcut的跨层连接对应的模块的索引
            identity_idx = (i + int(module_def['from'][0])) # shortcut的跨层连接对应的模块的索引
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)

        elif module_def['type'] == 'upsample':
            #上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)


    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def parse_module_defs2(module_defs):

    CBL_idx = [] # 包含BN层的卷积模块的索引列表（CBL: Conv-Bn-Leaky_relu）
    Conv_idx = [] # 不包含BN层的卷积模块的索引列表
    shortcut_idx=dict() # (key,value),其中key，value为一对生成一个shortcut模块的两个输入特征图的卷积模块，key为直连接，value为跨层连接
    shortcut_all=set() # 生成shortcut模块的输入特征图的卷积模块的索引集合，每个shortcut模块对应两个这样的卷积模块
    ignore_idx = set() # 不能够被剪枝的卷积模块的索引集合（spp前一个CBL不剪,上采样层前的卷积模块，...）
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            # if module_def['batch_normalize'] == '1':
            if module_def['batch_normalize'] == 1: 
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                #spp前一个CBL不剪 区分spp和tiny
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'route' and 'groups' in module_defs[i+1]:
                ignore_idx.add(i)

        elif module_def['type'] == 'upsample':
            #上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)

        elif module_def['type'] == 'shortcut':
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            # identity_idx = (i + int(module_def['from'])) # shortcut的跨层连接对应的模块的索引
            identity_idx = (i + int(module_def['from'][0])) # shortcut的跨层连接对应的模块的索引 from=-3
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

            if module_defs[identity_idx]['type'] == 'convolutional': # shortcut的跨层连接对应的模块为卷积模块
                #ignore_idx.add(identity_idx)
                shortcut_idx[i-1]=identity_idx # shortcut前一层的卷积模块和shortcut跨层连接对应的卷积模块构成一对(key,value)
                shortcut_all.add(identity_idx) # 将shortcut跨层连接对应的卷积模块添加入shortcut_all
            elif module_defs[identity_idx]['type'] == 'shortcut': # shortcut的跨层连接对应的模块为shortcut模块
                
                #ignore_idx.add(identity_idx - 1)
                shortcut_idx[i-1]=identity_idx-1 # shortcut前一层的卷积模块和“shortcut跨层连接对应的shortcut模块的前一层的卷积模块”构成一对(key,value)
                shortcut_all.add(identity_idx-1) # 将shortcut跨层连接对应的shortcut模块的前一层的卷积模块添加入shortcut_all
            shortcut_all.add(i-1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx] # 能够被剪枝的模块（包含BN层但是不在ignore_idx里的卷积模块）

    return CBL_idx, Conv_idx, prune_idx,shortcut_idx,shortcut_all



def parse_module_defs4(module_defs):

    CBL_idx = []
    Conv_idx = []
    shortcut_idx= [] # shortcut前一层的卷积模块
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            # if module_def['batch_normalize'] == '1':
            if module_def['batch_normalize'] == 1: 
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
        elif module_def['type'] == 'shortcut':
            shortcut_idx.append(i-1)

    return CBL_idx, Conv_idx, shortcut_idx




def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key == 'batch_normalize' and value == 0:
                    continue

                if key != 'type':
                    if key == 'anchors':
                        value = ', '.join(','.join(str(int(i)) for i in j) for j in value)

                    #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
                    if (key in ['from', 'layers', 'mask']) or (key == 'size' and not isinstance(value,int) ):
                        value = ', '.join(str(i) for i in value)
                    #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx, epoch, idx2mask=None, opt=None):
        if sr_flag:
            # s = s if epoch <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1
            if idx2mask:
                for idx in idx2mask:
                    bn_module = module_list[idx][1]
                    #bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight.data) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.data.sub_(0.99 * s * torch.sign(bn_module.weight.data) * idx2mask[idx].cuda())


def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []

        #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
        # for layer_i in module_defs[idx - 1]['layers'].split(","):
        for layer_i in module_defs[idx - 1]['layers']:
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

        if len(route_in_idxs) == 1:
            mask = CBLidx2mask[route_in_idxs[0]]
            if 'groups' in module_defs[idx - 1]:
                return mask[(mask.shape[0]//2):]
            return mask

        elif len(route_in_idxs) == 2:
            # return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
            if module_defs[route_in_idxs[0]]['type'] == 'upsample': 
                mask1 = CBLidx2mask[route_in_idxs[0] - 1]
            elif module_defs[route_in_idxs[0]]['type'] == 'convolutional':
                mask1 = CBLidx2mask[route_in_idxs[0]]
            if module_defs[route_in_idxs[1]]['type'] == 'convolutional':
                mask2 = CBLidx2mask[route_in_idxs[1]]
            else:
                mask2 = CBLidx2mask[route_in_idxs[1] - 1]
            return np.concatenate([mask1, mask2])

        elif len(route_in_idxs) == 4:
            #spp结构中最后一个route
            mask = CBLidx2mask[route_in_idxs[-1]]
            return np.concatenate([mask, mask, mask, mask])

        else:
            print("Something wrong with route module!")
            raise Exception
    elif module_defs[idx - 1]['type'] == 'maxpool':  #tiny
        if module_defs[idx - 2]['type'] == 'route':  #v4 tiny
            return get_input_mask(module_defs, idx - 1, CBLidx2mask)
        else:
            return CBLidx2mask[idx - 2]  #v3 tiny

def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()


def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    for idx in prune_idx:
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)

        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)

        for next_idx in next_idx_list:
            next_conv = pruned_model.module_list[next_idx][0]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                next_conv.bias.data.add_(offset)

        bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask



def update_activation(i, pruned_model, activation, CBL_idx):
    next_idx = i + 1
    if pruned_model.module_defs[next_idx]['type'] == 'convolutional':#下一个模块是CBL
        next_conv = pruned_model.module_list[next_idx][0]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        if next_idx in CBL_idx:#下一个模块是CBL
            next_bn = pruned_model.module_list[next_idx][1]
            next_bn.running_mean.data.sub_(offset) #bn层的running(moving)-cov(activation)
        else:
            next_conv.bias.data.add_(offset) #下一个是CL,cov(activation)加到cov的bias



def prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = torch.zeros(int(model_def['filters'])).cuda()
            if i in prune_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()#之前变成numpy了
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)#加mask
                if model_def['activation'] == 'leaky':#把bn的beta（bias）放入CBL的激活函数，输出，往下传递
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * bn_module.bias.data.mul(F.softplus(bn_module.bias.data).tanh())
                update_activation(i, pruned_model, activation, CBL_idx)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            # from_layer = int(model_def['from'])
            from_layer = int(model_def['from'][0])
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)
            


        elif model_def['type'] == 'route':
            #spp不参与剪枝，其中的route不用更新，仅占位
            #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
            # from_layers = [int(s) for s in model_def['layers'].split(',')]
            from_layers = model_def['layers']
            #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
                if 'groups' in model_def:
                    activation = activation[(activation.shape[0]//2):]                
                update_activation(i, pruned_model, activation, CBL_idx)
            elif len(from_layers) == 2:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]
                activation = torch.cat((actv1, actv2))
                update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
            activations.append(activations[i-1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':  #区分spp和tiny
            if model.module_defs[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i-1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)
       
    return pruned_model


def get_mask(model, prune_idx, shortcut_idx):
    sort_prune_idx=[idx for idx in prune_idx if idx not in shortcut_idx]
    bn_weights = gather_bn_weights(model.module_list, sort_prune_idx)
    sorted_bn = torch.sort(bn_weights)[0]
    highest_thre = []
    for idx in sort_prune_idx:
        #.item()可以得到张量里的元素值
        highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
    highest_thre = min(highest_thre)
    filters_mask = []
    idx_new=dict()
    #CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    for idx in prune_idx:
        bn_module = model.module_list[idx][1]        
        if idx not in shortcut_idx:
            mask = obtain_bn_mask(bn_module, torch.tensor(highest_thre)).cpu()
            idx_new[idx]=mask
        else:
            mask=idx_new[shortcut_idx[idx]]
            idx_new[idx]=mask        

        filters_mask.append(mask.clone())

    prune2mask = {idx: mask for idx, mask in zip(prune_idx, filters_mask)}
    return prune2mask
                    
def get_mask2(model, prune_idx, percent):
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    sorted_bn = torch.sort(bn_weights)[0]
    thre_index = int(len(sorted_bn) * percent)
    thre = sorted_bn[thre_index]

    filters_mask = []
    for idx in prune_idx:
        bn_module = model.module_list[idx][1]        
        mask = obtain_bn_mask(bn_module, thre).cpu()
        filters_mask.append(mask.clone())

    prune2mask = {idx: mask for idx, mask in zip(prune_idx, filters_mask)}
    return prune2mask
                    
def merge_mask(model, CBLidx2mask, CBLidx2filters):
    for i in range(len(model.module_defs) - 1, -1, -1): # 反向遍历模型的所有模块 len()-1,len()-2 ....0 共len个
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut': 
            if model.module_defs[i]['is_access']: #如果是true 就跳过下面代码，直接下一个循环
                continue

            Merge_masks =  []
            layer_i = i
            #while  shortcut的跨层还是shortcut，直到跨层是cov
            while mtype == 'shortcut': # 将shortcut的直连和跨层连接和多级跨层连接的卷积模块的剪枝掩码添加到Merge_masks列表里
                model.module_defs[layer_i]['is_access'] = True

                if model.module_defs[layer_i-1]['type'] == 'convolutional': 
                    bn = int(model.module_defs[layer_i-1]['batch_normalize'])
                    if bn: 
                        Merge_masks.append(CBLidx2mask[layer_i-1].unsqueeze(0))

                #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
                # layer_i = int(model.module_defs[layer_i]['from'])+layer_i 
                layer_i = int(model.module_defs[layer_i]['from'][0])+layer_i #跳转到from处
                #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#

                mtype = model.module_defs[layer_i]['type']#from处的层
                #若if通过，则与这个shortcut都关联需要共享mask的就找全了   这个cov就是用来调整下一块区域的通道
                if mtype == 'convolutional':    #对这个cov处理，并不再循环，否则循环到找到cov为止
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn: 
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))



                #shortcut直连层与跨层连接的shortcut(conv),通道数都是一样的  如size是[num]
            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)#list 拼成n*num的tensor
                """
                重点：
                    * 剪枝掩码对1求并集，即但凡有一个卷积模块中的某个位置的输出通道不被剪枝，则所有卷积模块在该位置的输出通道都不被剪枝
                    * 即剪枝掩码对0求交集,即只有所有卷积模块在某个位置的输出通道被剪枝，则所有卷积模块在该位置的输出通道才能被剪枝
                """
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float() #所有行对应位置相加，有一个1就只能全部保留
            else:
                merge_mask = Merge_masks[0].float()

            layer_i = i
            mtype = 'shortcut'#更新这一组相关的BN通道
            while mtype == 'shortcut':

                if model.module_defs[layer_i-1]['type'] == 'convolutional': 
                    bn = int(model.module_defs[layer_i-1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i-1] = merge_mask
                        CBLidx2filters[layer_i-1] = int(torch.sum(merge_mask).item())

                #+++++++++++++++++++++++++ insert +++++++++++++++++++++++++#
                # layer_i = int(model.module_defs[layer_i]['from'])+layer_i 
                layer_i = int(model.module_defs[layer_i]['from'][0])+layer_i 
                #+++++++++++++++++++++++++ insert end++++++++++++++++++++++#
                
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional': 
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:     
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())
