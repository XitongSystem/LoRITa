import copy, torch
import numpy as np
from .test_model import get_test_acc
from .vgg_USV import VGG_USV
from .resnet_USV import ResNet18_USV, ResNet20_USV
from .resnet_USV_image import ResNet18_USV_image

def uniform_decompose_vit(model, args):
    accs, Us, Ss, Vs = [], [], [], []

    for rank in range(args.dim, 1, -3):
        layer = 0
        for name, param in model.named_parameters():
            if 'qkv' in name or 'out' in name or 'net' in name and 'bias' not in name:
                
                if rank == args.dim:
                    U, S, V = torch.linalg.svd(param.data, full_matrices=False)
                    Us.append(U)
                    Ss.append(S)
                    Vs.append(V)

                S_copy = Ss[layer]
                S_copy[rank:] = 0
                param.data = Us[layer] @ torch.diag(S_copy) @ Vs[layer]

                layer += 1
    
        acc = get_test_acc(model, args)
        accs.append([rank, acc])
    return accs

def layer_decompose_vit(model, args):
    accs = []

    # attention
    UA, SA, SAr, VA = [], [], [], []

    # full-connected
    UL, SL, SLr, VL = [], [], [], []
    for name, param in model.named_parameters():

        if 'qkv' in name or 'out' in name:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            UA.append(U)
            SA.append(S)
            VA.append(V)
            SAr.append(S/torch.max(S))

        elif 'net' in name and 'bias' not in name:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            UL.append(U)
            SL.append(S)
            VL.append(V)
            SLr.append(S/torch.max(S))

    SLr = torch.cat(SL)
    sorted_SLr_index = torch.argsort(SLr)

    SAr = torch.cat(SA)
    sorted_SAr_index = torch.argsort(SAr)

    # ratio
    lenL = len(sorted_SLr_index)
    lenA = len(sorted_SAr_index)
    for rank in range(1, 100, 1):
        ratio = rank/100.0

        biasL, biasA, layerL, layerA = 0, 0, 0, 0
        for name, param in model.named_parameters():
            if 'qkv' in name or 'out' in name:
                num_S = len(SA[layerA])
                mask_element = torch.logical_and((sorted_SAr_index[:int(lenA*ratio)] - biasA >= 0),
                                                (sorted_SAr_index[:int(lenA*ratio)] < biasA + num_S))
                mask_element = sorted_SAr_index[:int(lenA*ratio)][mask_element] - biasA
                biasA += num_S

                S_copy = SA[layerA]
                S_copy[mask_element] = 0
                param.data = UA[layerA] @ torch.diag(S_copy) @ VA[layerA]

                layerA += 1

            elif 'net' in name and 'bias' not in name:
                num_S = len(SL[layerL])
                mask_element = torch.logical_and((sorted_SLr_index[:int(lenL*ratio)] - biasL >= 0),
                                                (sorted_SLr_index[:int(lenL*ratio)] < biasL + num_S))
                mask_element = sorted_SLr_index[:int(lenL*ratio)][mask_element] - biasL
                biasL += num_S

                S_copy = SL[layerL]
                S_copy[mask_element] = 0
                param.data = UL[layerL] @ torch.diag(S_copy) @ VL[layerL]

                layerL += 1
    
        acc = get_test_acc(model, args)
        accs.append([ratio, acc])
    return accs

def layer_decompose_fcn(model, args):
    accs, Us, Ss, Sr, Vs = [], [], [], [], []
    #d = 0
    for name, param in model.named_parameters():
        if 'net' in name and 'bias' not in name:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            #d += param.data.numel()
            Us.append(U)
            Ss.append(S)
            Vs.append(V)
            Sr.append(S/torch.max(S))
    #torch.save(Sr, 'FCN_'+str(args.factor)+'.pt')
    Sr = torch.cat(Ss)
    
    sorted_Sr_index = torch.argsort(Sr)
    #print(Sr[sorted_Sr_index[0]],Sr[sorted_Sr_index[-1]]) #small-->large

    for rank in range(100, len(sorted_Sr_index), 5):
        bias, layer = 0, 0

        for name, param in model.named_parameters():
            if 'qkv' in name or 'out' in name or 'net' in name and 'bias' not in name:

                num_S = len(Ss[layer])
                mask_element = torch.logical_and((sorted_Sr_index[:rank] - bias >= 0),
                                                (sorted_Sr_index[:rank] < bias + num_S))
                mask_element = sorted_Sr_index[:rank][mask_element] - bias
                bias += num_S

                S_copy = Ss[layer]
                S_copy[mask_element] = 0
                param.data = Us[layer] @ torch.diag(S_copy) @ Vs[layer]

                layer += 1
    
        acc = get_test_acc(model, args)
        accs.append([rank, acc])
    return accs


def uniform_decompose_cnn(model, args):
    accs, Us, Ss, Vs = [], [], [], []
    UL, SL, VL = [], [], []

    acc = args.reference_acc
    for rank in range(100, 1, -3):
        ratio = rank/100

        layer, layerL = 0, 0
        for name, param in model.named_parameters():

            # conv weight
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                orig_shape = param.data.shape

                if rank == 100:
                    U, S, V = torch.linalg.svd(param.data.view(param.data.shape[0],-1), full_matrices=False)
                    Us.append(U)
                    Ss.append(S)
                    Vs.append(V)

                S_copy = Ss[layer]
                S_copy[int(len(S_copy)*ratio):] = 0
                param.data = (Us[layer] @ torch.diag(S_copy) @ Vs[layer]).view(orig_shape)

                layer += 1
            elif len(param.data.shape) > 1:
                if rank == 100:
                    U, S, V = torch.linalg.svd(param.data, full_matrices=False)
                    UL.append(U)
                    SL.append(S)
                    VL.append(V)

                S_copy = SL[layerL]
                S_copy[int(len(S_copy)*ratio):] = 0
                param.data = (UL[layerL] @ torch.diag(S_copy) @ VL[layerL])

                layerL += 1

    
        acc = get_test_acc(model, args)
        accs.append([rank, acc])
    return accs

def layer_decompose_cnn(model, args):
    accs, UC, SC, SCr, VC = [], [], [], [], []
    UL, SL, SLr, VL = [], [], [], []

    # decomposition
    for name, param in model.named_parameters():
        # conv weight
        if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
            orig_shape = param.data.shape
            U, S, V = torch.linalg.svd(param.data.view(param.data.shape[0],-1), full_matrices=False)
            UC.append(U)
            SC.append(S)
            VC.append(V)
            SCr.append(S/torch.max(S))

        elif len(param.data.shape) > 1:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            UL.append(U)
            SL.append(S)
            VL.append(V)
            SLr.append(S/torch.max(S))

    SLr = torch.cat(SL)
    sorted_SLr_index = torch.argsort(SLr)

    SCr = torch.cat(SCr)
    sorted_SCr_index = torch.argsort(SCr)

    # ratio
    lenL = len(sorted_SLr_index)
    lenC = len(sorted_SCr_index)

    # model used for finetuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    if args.dataset.lower() == 'cifar100':
        num_classes = 100
    # compression based on S
    tor = 0
    acc = args.reference_acc
    for rank in range(1, 100, 1):
        ratio = rank/100.0

        # reinit model
        if args.model.lower() == 'resnet18':
            model_USV = ResNet18_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'resnet20':
            model_USV = ResNet20_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'vgg13':
            model_USV = VGG_USV('VGG13',num_classes=num_classes).to(device) 
        elif args.model.lower() == 'vgg16':
            model_USV = VGG_USV('VGG16',num_classes=num_classes).to(device) 
        model_USV_dict = {} 
        for name, param in model_USV.named_parameters():
            model_USV_dict[name] = param.data


        biasL, biasC, layerL, layerC = 0, 0, 0, 0
        original_size = 0.0
        compress_size = 0.0
        ranks = []

        model_prev = copy.deepcopy(model)
        for name, param in model.named_parameters():
            orig_shape = param.data.shape

            original_size += param.data.numel()
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                num_S = len(SC[layerC])
                mask_element = torch.logical_and((sorted_SCr_index[:int(lenC*ratio)] - biasC >= 0),
                                                (sorted_SCr_index[:int(lenC*ratio)] < biasC + num_S))
                mask_element = sorted_SCr_index[:int(lenC*ratio)][mask_element] - biasC
                biasC += num_S

                S_copy = SC[layerC]
                S_copy[mask_element] = 0
                param.data = (UC[layerC] @ torch.diag(S_copy) @ VC[layerC]).view(orig_shape)

                prefix = '.'.join(name.split('.')[:-1])
                model_USV_dict[prefix + '.U'] = UC[layerC][:,S_copy>0]
                model_USV_dict[prefix + '.V'] = VC[layerC][S_copy>0,:]
                model_USV_dict[prefix + '.S'] = S_copy[S_copy>0]

                layerC += 1

                compress_size += model_USV_dict[prefix + '.U'].numel() + model_USV_dict[prefix + '.V'].numel()
                ranks.append(model_USV_dict[prefix + '.S'].numel())
            elif len(param.data.shape) > 1:
                num_S = len(SL[layerL])
                mask_element = torch.logical_and((sorted_SLr_index[:int(lenL*ratio)] - biasL >= 0),
                                                (sorted_SLr_index[:int(lenL*ratio)] < biasL + num_S))
                mask_element = sorted_SLr_index[:int(lenL*ratio)][mask_element] - biasL
                biasL += num_S

                S_copy = SL[layerL]
                if args.compress_fc:
                    S_copy[mask_element] = 0
                param.data = UL[layerL] @ torch.diag(S_copy) @ VL[layerL]

                prefix = '.'.join(name.split('.')[:-1])
                model_USV_dict[prefix + '.U'] = UL[layerL][:,S_copy>0]
                model_USV_dict[prefix + '.V'] = VL[layerL][S_copy>0,:]
                model_USV_dict[prefix + '.S'] = S_copy[S_copy>0]

                layerL += 1

                compress_size += model_USV_dict[prefix + '.U'].numel() + model_USV_dict[prefix + '.V'].numel()
                ranks.append(model_USV_dict[prefix + '.S'].numel())
            else:
                model_USV_dict[name] = param.data
                compress_size += param.data.numel()

        # setup weight
        for name, param in model_USV.named_parameters():
            param.data = model_USV_dict[name]

        # setup batchnorm states
        batchnorm_state = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                batchnorm_state[name] = {}
                batchnorm_state[name]['m'] = module.running_mean  
                batchnorm_state[name]['v'] = module.running_var

        for name, module in model_USV.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = batchnorm_state[name]['m'] 
                module.running_var = batchnorm_state[name]['v'] 

        acc = get_test_acc(model_USV, args)
        accs.append([ratio, acc, compress_size/original_size, ranks])

        if args.reference_acc > (acc + 0.01):
            tor += 1
        else:
            tor = 0 
        if tor > 3:
            break

        model_prev = copy.deepcopy(model)

    return accs, model_prev


def layer_decompose_cnn_iterative(model, args, grid=2, starting_rank=1, rank_threshold=2):
    accs = []
    # model used for finetuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    if args.dataset.lower() == 'cifar100':
        num_classes = 100
    # compression based on S
    tor = 0
    acc = args.reference_acc
    for rank in np.arange(starting_rank, 100, grid):
        ratio = rank/100.0

        # reinit model
        if args.model.lower() == 'resnet18':
            model_USV = ResNet18_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'resnet20':
            model_USV = ResNet20_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'vgg13':
            model_USV = VGG_USV('VGG13',num_classes=num_classes).to(device) 
        elif args.model.lower() == 'vgg16':
            model_USV = VGG_USV('VGG16',num_classes=num_classes).to(device) 
        model_USV_dict = {} 
        for name, param in model_USV.named_parameters():
            model_USV_dict[name] = param.data

        # SVD
        UC, SC, SCr, VC = [], [], [], []
        UL, SL, SLr, VL = [], [], [], []

        # decomposition
        for name, param in model.named_parameters():
            # conv weight
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                orig_shape = param.data.shape
                U, S, V = torch.linalg.svd(param.data.view(param.data.shape[0],-1), full_matrices=False)
                UC.append(U)
                SC.append(S)
                VC.append(V)
                SCr.append(S/torch.max(S))

            elif len(param.data.shape) > 1:
                U, S, V = torch.linalg.svd(param.data, full_matrices=False)
                UL.append(U)
                SL.append(S)
                VL.append(V)
                SLr.append(S/torch.max(S))

        SLr = torch.cat(SL)
        sorted_SLr_index = torch.argsort(SLr)

        SCr = torch.cat(SCr)
        #torch.save(SCr,"SCr.pt")
        sorted_SCr_index = torch.argsort(SCr)

        # ratio
        lenL = len(sorted_SLr_index)
        lenC = len(sorted_SCr_index)

        biasL, biasC, layerL, layerC = 0, 0, 0, 0
        original_size = 0.0
        compress_size = 0.0
        ranks = []

        model_prev = copy.deepcopy(model)
        for name, param in model.named_parameters():
            orig_shape = param.data.shape

            original_size += param.data.numel()
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                num_S = len(SC[layerC])
                mask_element = torch.logical_and((sorted_SCr_index[:int(lenC*ratio)] - biasC >= 0),
                                                (sorted_SCr_index[:int(lenC*ratio)] < biasC + num_S))
                mask_element = sorted_SCr_index[:int(lenC*ratio)][mask_element] - biasC
                biasC += num_S

                S_copy = SC[layerC]
                S_copy[mask_element] = 0
                if num_S - len(mask_element) <= rank_threshold:
                    S_copy[:rank_threshold] = SC[layerC][:rank_threshold]

                param.data = (UC[layerC] @ torch.diag(S_copy) @ VC[layerC]).view(orig_shape)

                prefix = '.'.join(name.split('.')[:-1])
                model_USV_dict[prefix + '.U'] = UC[layerC][:,S_copy>0]
                model_USV_dict[prefix + '.V'] = VC[layerC][S_copy>0,:]
                model_USV_dict[prefix + '.S'] = S_copy[S_copy>0]

                layerC += 1

                compress_size += model_USV_dict[prefix + '.U'].numel() + model_USV_dict[prefix + '.V'].numel()
                ranks.append(model_USV_dict[prefix + '.S'].numel())
            elif len(param.data.shape) > 1:
                num_S = len(SL[layerL])
                mask_element = torch.logical_and((sorted_SLr_index[:int(lenL*ratio)] - biasL >= 0),
                                                (sorted_SLr_index[:int(lenL*ratio)] < biasL + num_S))
                mask_element = sorted_SLr_index[:int(lenL*ratio)][mask_element] - biasL
                biasL += num_S

                S_copy = SL[layerL]
                if args.compress_fc:
                    S_copy[mask_element] = 0
                if num_S - len(mask_element) <= rank_threshold:
                    S_copy[:rank_threshold] = SL[layerL][:rank_threshold]

                param.data = UL[layerL] @ torch.diag(S_copy) @ VL[layerL]

                prefix = '.'.join(name.split('.')[:-1])
                model_USV_dict[prefix + '.U'] = UL[layerL][:,S_copy>0]
                model_USV_dict[prefix + '.V'] = VL[layerL][S_copy>0,:]
                model_USV_dict[prefix + '.S'] = S_copy[S_copy>0]

                layerL += 1

                compress_size += model_USV_dict[prefix + '.U'].numel() + model_USV_dict[prefix + '.V'].numel()
                ranks.append(model_USV_dict[prefix + '.S'].numel())
            else:
                model_USV_dict[name] = param.data
                compress_size += param.data.numel()
        
        # setup weight
        for name, param in model_USV.named_parameters():
            param.data = model_USV_dict[name]

        # setup batchnorm states
        batchnorm_state = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                batchnorm_state[name] = {}
                batchnorm_state[name]['m'] = module.running_mean  
                batchnorm_state[name]['v'] = module.running_var

        for name, module in model_USV.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = batchnorm_state[name]['m'] 
                module.running_var = batchnorm_state[name]['v'] 

        # finetune
        acc = get_test_acc(model_USV, args, 1e-5, True)
        accs.append([ratio, acc, compress_size/original_size, ranks])
        print(ranks, compress_size/original_size, acc)

        # set model back from model_USV
        for name, param in model_USV.named_parameters():
            model_USV_dict[name] = param.data 
        for name, param in model.named_parameters():
            orig_shape = param.data.shape
            original_size += param.data.numel()
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                prefix = '.'.join(name.split('.')[:-1])
                param.data = (model_USV_dict[prefix + '.U'] @ torch.diag(model_USV_dict[prefix + '.S']) @ model_USV_dict[prefix + '.V']).view(orig_shape)

            elif len(param.data.shape) > 1:
                prefix = '.'.join(name.split('.')[:-1])
                param.data = model_USV_dict[prefix + '.U'] @ torch.diag(model_USV_dict[prefix + '.S']) @ model_USV_dict[prefix + '.V']
            else:
                param.data = model_USV_dict[name]
                compress_size += param.data.numel()

        # setup batchnorm states
        batchnorm_state = {}
        for name, module in model_USV.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                batchnorm_state[name] = {}
                batchnorm_state[name]['m'] = module.running_mean  
                batchnorm_state[name]['v'] = module.running_var

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = batchnorm_state[name]['m'] 
                module.running_var = batchnorm_state[name]['v']

        if args.reference_acc > (acc + 0.01):
            tor += 1
        else:
            tor = 0 
            compression_checker = copy.deepcopy(model)
            best_rank = rank
        if tor > 3:
            break

    return accs, compression_checker, best_rank

def layer_decompose_cnn_iterative_layer(model, args, grid=2, best_ranks=[]):
    def reinit(args):
        # model used for finetuning
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = 10
        if args.dataset.lower() == 'cifar100':
            num_classes = 100

        # reinit model
        if args.model.lower() == 'resnet18':
            model_USV = ResNet18_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'resnet20':
            model_USV = ResNet20_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'vgg13':
            model_USV = VGG_USV('VGG13',num_classes=num_classes).to(device) 
        elif args.model.lower() == 'vgg16':
            model_USV = VGG_USV('VGG16',num_classes=num_classes).to(device) 

        return model_USV

    accs = []
    # compression based on S
    tor = 0
    acc = args.reference_acc
    compression_checker = model

    # SVD
    UC, SC, VC = [], [], []
    UL, SL, VL = [], [], []
    model_USV_dict = {}
    # decomposition
    layer = 0
    for name, param in model.named_parameters():

        prefix = '.'.join(name.split('.')[:-1])
        # conv weight
        if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
            orig_shape = param.data.shape
            U, S, V = torch.linalg.svd(param.data.view(param.data.shape[0],-1), full_matrices=False)
            SC.append(S)
            UC.append(U)
            VC.append(V)

            model_USV_dict[prefix + '.U'] = U
            model_USV_dict[prefix + '.S'] = S
            model_USV_dict[prefix + '.V'] = V

        elif len(param.data.shape) > 1:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            SL.append(S)
            UL.append(U)
            VL.append(V)

            model_USV_dict[prefix + '.U'] = U
            model_USV_dict[prefix + '.S'] = S
            model_USV_dict[prefix + '.V'] = V
        else:
            model_USV_dict[name] = param.data

    cur_ranks = best_ranks
    truncate = 0
    prev_model_USV = copy.deepcopy(model_USV_dict)
    while tor < 4:
        layer2compress = ''
        layer2compress_acc = 0
        layerL, layerC = 0, 0
        for name, param in model.named_parameters():
            model_USV = reinit(args)
            orig_shape = param.data.shape
            prefix = '.'.join(name.split('.')[:-1])
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                S_copy = copy.deepcopy(SC[layerC])
                if len(cur_ranks) > 0:
                    truncate = cur_ranks[layerC + layerL]
                S_copy[truncate-grid:] = 0
                if (S_copy>0).sum() > 0:
                    model_USV_dict[prefix + '.U'] = UC[layerC][:,S_copy>0]
                    model_USV_dict[prefix + '.V'] = VC[layerC][S_copy>0,:]
                    model_USV_dict[prefix + '.S'] = S_copy[S_copy>0]
                    compressed = True
                else:
                    compressed = False
                layerC += 1
            elif len(param.data.shape) > 1:
                S_copy = copy.deepcopy(SL[layerL])
                if args.compress_fc:
                    if len(cur_ranks) > 0:
                        truncate = cur_ranks[layerC + layerL]
                    S_copy[truncate-grid:] = 0

                if args.compress_fc:
                    if (S_copy>0).sum() > 0:
                        model_USV_dict[prefix + '.U'] = UL[layerL][:,S_copy>0]
                        model_USV_dict[prefix + '.V'] = VL[layerL][S_copy>0,:]
                        model_USV_dict[prefix + '.S'] = S_copy[S_copy>0]
                        compressed = True
                    else:
                        compressed = False
                else:
                    compressed = False
                layerL += 1
            else:
                model_USV_dict[name] = param.data
                compressed = False

            if compressed:
                print(prefix)
                check_layer = prefix
                # setup weight
                for name, param in model_USV.named_parameters():
                    prefix = '.'.join(name.split('.')[:-1])
                    if prefix != check_layer:
                        param.data = prev_model_USV[name]
                    else:
                        param.data = model_USV_dict[name]

                # setup batchnorm states
                batchnorm_state = {}
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        batchnorm_state[name] = {}
                        batchnorm_state[name]['m'] = module.running_mean  
                        batchnorm_state[name]['v'] = module.running_var

                for name, module in model_USV.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.running_mean = batchnorm_state[name]['m'] 
                        module.running_var = batchnorm_state[name]['v'] 

                # finetune
                acc = get_test_acc(model_USV, args, 1e-5, True, ratio=args.finetune_ratio)
                
                if acc > layer2compress_acc:
                    layer2compress = check_layer
                    layer2compress_acc = acc

        model_USV = reinit(args)
        print("layer2compress", layer2compress)
        # compress based on checker
        ranks = []
        layerL, layerC = 0, 0
        if layer2compress != '':
            for name, param in model.named_parameters():
                prefix = '.'.join(name.split('.')[:-1])
                truncate = 0
                # conv weight
                if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                    if len(cur_ranks) > 0:
                        truncate = cur_ranks[layerC + layerL]
                        if prefix == layer2compress:
                            truncate -= grid
                    
                    model_USV_dict[prefix + '.U'] = UC[layerC][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VC[layerC][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SC[layerC][:truncate]
                    ranks.append(truncate)
                    layerC += 1
                elif len(param.data.shape) > 1:
                    if len(cur_ranks) > 0:
                        truncate = cur_ranks[layerC + layerL]
                        if prefix == layer2compress:
                            truncate -= grid

                    model_USV_dict[prefix + '.U'] = UL[layerL][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VL[layerL][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SL[layerL][:truncate]
                    ranks.append(truncate)
                    layerL += 1
                else:
                    model_USV_dict[name] = param.data
        else:
            break

        for name, param in model_USV.named_parameters():
            param.data = model_USV_dict[name]

        for name, param in model.named_parameters():
            orig_shape = param.data.shape
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                prefix = '.'.join(name.split('.')[:-1])
                param.data = (model_USV_dict[prefix + '.U'] @ torch.diag(model_USV_dict[prefix + '.S']) @ model_USV_dict[prefix + '.V']).view(orig_shape)

            elif len(param.data.shape) > 1:
                prefix = '.'.join(name.split('.')[:-1])
                param.data = model_USV_dict[prefix + '.U'] @ torch.diag(model_USV_dict[prefix + '.S']) @ model_USV_dict[prefix + '.V']
            else:
                param.data = model_USV_dict[name]

        # setup batchnorm states
        batchnorm_state = {}
        for name, module in model_USV.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                batchnorm_state[name] = {}
                batchnorm_state[name]['m'] = module.running_mean  
                batchnorm_state[name]['v'] = module.running_var

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = batchnorm_state[name]['m'] 
                module.running_var = batchnorm_state[name]['v']

        layer2compress_acc = get_test_acc(model_USV, args, 1e-5, True, ratio=1.0)
        accs.append([ranks, layer2compress_acc])
        print('Pruned:', ranks, layer2compress_acc)
        if args.reference_acc > (layer2compress_acc + 0.01):
            tor += 1
        else:
            tor = 0 
            compression_checker = copy.deepcopy(model)
            best_ranks = ranks
        cur_ranks = ranks

    return accs, compression_checker, best_ranks

def layer_decompose_cnn_training(model, args, grid=2, best_ranks=[]):
    '''r
    model: pytorch model
    args: args of post_cnns.py
    grid: truncation grid
    best_ranks: a list inludes the known current best ranks per layer to be compressed
    '''
    def reinit(args):
        # model used for finetuning, it should be reinitialized whenever the rank changes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = 10
        if args.dataset.lower() == 'cifar100':
            num_classes = 100

        # reinit model
        if args.model.lower() == 'resnet18':
            model_USV = ResNet18_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'resnet20':
            model_USV = ResNet20_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'vgg13':
            model_USV = VGG_USV('VGG13',num_classes=num_classes).to(device) 
        elif args.model.lower() == 'vgg16':
            model_USV = VGG_USV('VGG16',num_classes=num_classes).to(device) 

        return model_USV

    accs = []
    tor = 0
    acc = args.reference_acc

    UC, SC, VC = [], [], []
    UL, SL, VL = [], [], []
    model_USV_dict = {}
    cur_ranks = []
    if len(best_ranks) > 0:
        cur_ranks = best_ranks

    layerC, layerL = 0, 0
    # SVD
    for name, param in model.named_parameters():

        prefix = '.'.join(name.split('.')[:-1])

        truncate = -1
        # conv weight
        if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
            orig_shape = param.data.shape
            U, S, V = torch.linalg.svd(param.data.view(param.data.shape[0],-1), full_matrices=False)
            SC.append(S)
            UC.append(U)
            VC.append(V)

            if len(best_ranks) > 0:
                truncate = cur_ranks[layerC + layerL]
                model_USV_dict[prefix + '.U'] = U[:,:truncate]
                model_USV_dict[prefix + '.S'] = S[:truncate]
                model_USV_dict[prefix + '.V'] = V[:truncate,:]
                #cur_ranks.append((S[:truncate]>0).sum().cpu().item())
            else:
                model_USV_dict[prefix + '.U'] = U
                model_USV_dict[prefix + '.S'] = S
                model_USV_dict[prefix + '.V'] = V
                cur_ranks.append((S>0).sum().cpu().item())
            layerC += 1

        elif len(param.data.shape) > 1:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            SL.append(S)
            UL.append(U)
            VL.append(V)

            if len(best_ranks) > 0:
                truncate = cur_ranks[layerC + layerL]
                model_USV_dict[prefix + '.U'] = U[:,:truncate]
                model_USV_dict[prefix + '.S'] = S[:truncate]
                model_USV_dict[prefix + '.V'] = V[:truncate,:]
                #cur_ranks.append((S[:truncate]>0).sum().cpu().item())
            else:
                model_USV_dict[prefix + '.U'] = U
                model_USV_dict[prefix + '.S'] = S
                model_USV_dict[prefix + '.V'] = V
                cur_ranks.append((S>0).sum().cpu().item())
            layerL += 1
        else:
            model_USV_dict[name] = param.data

    # save the model dict before compression
    truncate = 0
    prev_model_USV = copy.deepcopy(model_USV_dict)

    # setup batchnorm states
    batchnorm_state = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            batchnorm_state[name] = {}
            batchnorm_state[name]['m'] = module.running_mean  
            batchnorm_state[name]['v'] = module.running_var

    while tor < 4:
        layer2compress = ''
        layer2compress_loss = 1000
        layerL, layerC = 0, 0

        # compression
        for name, param in model.named_parameters():
            model_USV = reinit(args)
            orig_shape = param.data.shape
            prefix = '.'.join(name.split('.')[:-1])
            compressed = False

            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                truncate = cur_ranks[layerC + layerL]
                truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                if truncate > 1:
                    model_USV_dict[prefix + '.U'] = UC[layerC][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VC[layerC][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SC[layerC][:truncate]
                    compressed = True
                layerC += 1
            elif len(param.data.shape) > 1:
                truncate = cur_ranks[layerC + layerL]
                truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                if args.compress_fc:
                    if truncate > 1:
                        model_USV_dict[prefix + '.U'] = UL[layerL][:,:truncate]
                        model_USV_dict[prefix + '.V'] = VL[layerL][:truncate,:]
                        model_USV_dict[prefix + '.S'] = SL[layerL][:truncate]
                        compressed = True
                layerL += 1
            else:
                model_USV_dict[name] = param.data

            if compressed:
                print(prefix)
                check_layer = prefix
                # setup weight
                for name, param in model_USV.named_parameters():
                    prefix = '.'.join(name.split('.')[:-1])
                    if prefix != check_layer:
                        param.data = prev_model_USV[name]
                    else:
                        param.data = model_USV_dict[name]

                # setup batch norm state
                for name, module in model_USV.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.running_mean = batchnorm_state[name]['m'] 
                        module.running_var = batchnorm_state[name]['v'] 

                # no finetune
                args.no_aug = True
                args.finetune = False

                # select ranks based on the training loss, there is no training at this step
                # set ratio = 0.9999 to avoid data augmentation on training dataset 
                acc, loss = get_test_acc(model_USV, args, ratio=0.9999, testing=False)
                
                if loss < layer2compress_loss:
                    layer2compress = check_layer
                    layer2compress_loss = loss

        # model should be reinitialized whenever
        # the rank changes
        model_USV = reinit(args)
        print("layer2compress", layer2compress)
        # compress based on checker
        ranks = []
        layerL, layerC = 0, 0
        if layer2compress != '':
            for name, param in model.named_parameters():
                prefix = '.'.join(name.split('.')[:-1])

                # conv weight
                if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):       
                    truncate = cur_ranks[layerC + layerL]
                    if prefix == layer2compress:
                        truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                    model_USV_dict[prefix + '.U'] = UC[layerC][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VC[layerC][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SC[layerC][:truncate]
                    ranks.append(truncate)
                    layerC += 1
                elif len(param.data.shape) > 1:
                    truncate = cur_ranks[layerC + layerL]
                    if prefix == layer2compress:
                        truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                    model_USV_dict[prefix + '.U'] = UL[layerL][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VL[layerL][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SL[layerL][:truncate]
                    ranks.append(truncate)
                    layerL += 1
                else:
                    model_USV_dict[name] = param.data
        else:
            break
        
        # set the model after compression
        for name, param in model_USV.named_parameters():
            param.data = model_USV_dict[name].detach().clone()
        for name, module in model_USV.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = batchnorm_state[name]['m'] 
                module.running_var = batchnorm_state[name]['v'] 

        # finetune
        args.no_aug = False # using data augmentation in finetuning
        args.finetune = True
        layer2compress_acc = get_test_acc(model_USV, args, 1e-5, True, ratio=1.0, testing=True)
        accs.append([ranks, layer2compress_acc])
        print('Pruned:', ranks, layer2compress_acc)
        if args.reference_acc > (layer2compress_acc + 0.01):
            tor += 1
        else:
            # save intermediate results
            tor = 0 
            best_ranks = ranks
            batchnorm_state_checker = {}
            model_checker = copy.deepcopy(model_USV_dict)
            for name, module in model_USV.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    batchnorm_state_checker[name] = {}
                    batchnorm_state_checker[name]['m'] = module.running_mean
                    batchnorm_state_checker[name]['v'] = module.running_var
        cur_ranks = ranks

        # update saved model for the next round
        prev_model_USV = copy.deepcopy(model_USV_dict)

    try:
        # set state back to the original model
        for name, param in model.named_parameters():
            orig_shape = param.data.shape
            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                prefix = '.'.join(name.split('.')[:-1])
                param.data = (model_checker[prefix + '.U'] @ torch.diag(model_checker[prefix + '.S']) @ model_checker[prefix + '.V']).view(orig_shape)
            elif len(param.data.shape) > 1:
                prefix = '.'.join(name.split('.')[:-1])
                param.data = model_checker[prefix + '.U'] @ torch.diag(model_checker[prefix + '.S']) @ model_checker[prefix + '.V']
            else:
                param.data = model_checker[name]
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = batchnorm_state_checker[name]['m'] 
                module.running_var  = batchnorm_state_checker[name]['v'] 
    except:
        pass
        
    return accs, best_ranks

def layer_decompose_cnn_training2target(model, args, grid=2, best_ranks=[], target=1.0):
    def reinit(args):
        # model used for finetuning
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = 10
        if args.dataset.lower() == 'cifar100':
            num_classes = 100

        # reinit model
        if args.model.lower() == 'resnet18':
            model_USV = ResNet18_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'resnet20':
            model_USV = ResNet20_USV(num_classes=num_classes).to(device)
        elif args.model.lower() == 'vgg13':
            model_USV = VGG_USV('VGG13',num_classes=num_classes).to(device) 
        elif args.model.lower() == 'vgg16':
            model_USV = VGG_USV('VGG16',num_classes=num_classes).to(device) 

        return model_USV

    def num_of_params(model):
        num = 0
        for param in model.parameters():
            num += param.numel()
        return num

    sum_num = num_of_params(model)

    accs = []
    # compression based on S
    tor = 0
    acc = args.reference_acc

    # SVD
    UC, SC, VC = [], [], []
    UL, SL, VL = [], [], []
    model_USV_dict = {}
    # decomposition
    layer = 0

    cur_ranks = []
    if len(best_ranks) > 0:
        cur_ranks = best_ranks

    layerC, layerL = 0, 0
    for name, param in model.named_parameters():

        prefix = '.'.join(name.split('.')[:-1])

        truncate = -1
        # conv weight
        if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
            orig_shape = param.data.shape
            U, S, V = torch.linalg.svd(param.data.view(param.data.shape[0],-1), full_matrices=False)
            SC.append(S)
            UC.append(U)
            VC.append(V)

            if len(best_ranks) > 0:
                truncate = cur_ranks[layerC + layerL]
                model_USV_dict[prefix + '.U'] = U[:,:truncate]
                model_USV_dict[prefix + '.S'] = S[:truncate]
                model_USV_dict[prefix + '.V'] = V[:truncate,:]
                #cur_ranks.append((S[:truncate]>0).sum().cpu().item())
            else:
                model_USV_dict[prefix + '.U'] = U
                model_USV_dict[prefix + '.S'] = S
                model_USV_dict[prefix + '.V'] = V
                cur_ranks.append((S>0).sum().cpu().item())
            layerC += 1

        elif len(param.data.shape) > 1:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            SL.append(S)
            UL.append(U)
            VL.append(V)

            if len(best_ranks) > 0:
                truncate = cur_ranks[layerC + layerL]
                model_USV_dict[prefix + '.U'] = U[:,:truncate]
                model_USV_dict[prefix + '.S'] = S[:truncate]
                model_USV_dict[prefix + '.V'] = V[:truncate,:]
                #cur_ranks.append((S[:truncate]>0).sum().cpu().item())
            else:
                model_USV_dict[prefix + '.U'] = U
                model_USV_dict[prefix + '.S'] = S
                model_USV_dict[prefix + '.V'] = V
                cur_ranks.append((S>0).sum().cpu().item())
            layerL += 1
        else:
            model_USV_dict[name] = param.data

    truncate = 0
    prev_model_USV = copy.deepcopy(model_USV_dict)

    # setup batchnorm states
    batchnorm_state = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            batchnorm_state[name] = {}
            batchnorm_state[name]['m'] = module.running_mean  
            batchnorm_state[name]['v'] = module.running_var

    while tor < 4:
        layer2compress = ''
        layer2compress_loss = 1000
        layerL, layerC = 0, 0
        for name, param in model.named_parameters():
            model_USV = reinit(args)
            orig_shape = param.data.shape
            prefix = '.'.join(name.split('.')[:-1])
            compressed = False

            if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
                truncate = cur_ranks[layerC + layerL]
                truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                if truncate > 1:
                    model_USV_dict[prefix + '.U'] = UC[layerC][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VC[layerC][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SC[layerC][:truncate]
                    compressed = True
                layerC += 1
            elif len(param.data.shape) > 1:
                truncate = cur_ranks[layerC + layerL]
                truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                if args.compress_fc:
                    if truncate > 1:
                        model_USV_dict[prefix + '.U'] = UL[layerL][:,:truncate]
                        model_USV_dict[prefix + '.V'] = VL[layerL][:truncate,:]
                        model_USV_dict[prefix + '.S'] = SL[layerL][:truncate]
                        compressed = True
                layerL += 1
            else:
                model_USV_dict[name] = param.data

            if compressed:
                print(prefix)
                check_layer = prefix
                # setup weight
                for name, param in model_USV.named_parameters():
                    prefix = '.'.join(name.split('.')[:-1])
                    if prefix != check_layer:
                        param.data = prev_model_USV[name]
                    else:
                        param.data = model_USV_dict[name]

                for name, module in model_USV.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.running_mean = batchnorm_state[name]['m'] 
                        module.running_var = batchnorm_state[name]['v'] 

                # no finetune
                args.no_aug = True
                args.finetune = False

                acc, loss = get_test_acc(model_USV, args, ratio=0.9999, testing=False)
                
                if loss < layer2compress_loss:
                    layer2compress = check_layer
                    layer2compress_loss = loss

        model_USV = reinit(args)
        print("layer2compress", layer2compress)
        # compress based on checker
        ranks = []
        layerL, layerC = 0, 0
        if layer2compress != '':
            for name, param in model.named_parameters():
                prefix = '.'.join(name.split('.')[:-1])

                # conv weight
                if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):       
                    truncate = cur_ranks[layerC + layerL]
                    if prefix == layer2compress:
                        truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                    model_USV_dict[prefix + '.U'] = UC[layerC][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VC[layerC][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SC[layerC][:truncate]
                    ranks.append(truncate)
                    layerC += 1
                elif len(param.data.shape) > 1:
                    truncate = cur_ranks[layerC + layerL]
                    if prefix == layer2compress:
                        truncate = (truncate - grid) if (truncate - grid) > 0 else 1

                    model_USV_dict[prefix + '.U'] = UL[layerL][:,:truncate]
                    model_USV_dict[prefix + '.V'] = VL[layerL][:truncate,:]
                    model_USV_dict[prefix + '.S'] = SL[layerL][:truncate]
                    ranks.append(truncate)
                    layerL += 1
                else:
                    model_USV_dict[name] = param.data
        else:
            break

        for name, param in model_USV.named_parameters():
            param.data = model_USV_dict[name]

        # compute ratio
        sum_num_usv = num_of_params(model_USV)
        for name, module in model_USV.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = batchnorm_state[name]['m'] 
                module.running_var = batchnorm_state[name]['v'] 

        # no finetune
        args.no_aug = False
        args.finetune = True

        layer2compress_acc = 100.0
        if sum_num_usv/sum_num <= target:
            layer2compress_acc = get_test_acc(model_USV, args, 1e-5, True, ratio=1.0, testing=True)
        accs.append([ranks, layer2compress_acc])
        print('Pruned:', ranks, layer2compress_acc)
        if args.reference_acc > (layer2compress_acc + 0.01):
            tor += 1
        else:
            tor = 0 
            best_ranks = ranks
            batchnorm_state_checker = {}
            model_checker = copy.deepcopy(model_USV_dict)
            for name, module in model_USV.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    batchnorm_state_checker[name] = {}
                    batchnorm_state_checker[name]['m'] = module.running_mean
                    batchnorm_state_checker[name]['v'] = module.running_var
        cur_ranks = ranks
        prev_model_USV = copy.deepcopy(model_USV_dict)

    # set state back to model
    for name, param in model.named_parameters():
        orig_shape = param.data.shape
        if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
            prefix = '.'.join(name.split('.')[:-1])
            param.data = (model_checker[prefix + '.U'] @ torch.diag(model_checker[prefix + '.S']) @ model_checker[prefix + '.V']).view(orig_shape)
        elif len(param.data.shape) > 1:
            prefix = '.'.join(name.split('.')[:-1])
            param.data = model_checker[prefix + '.U'] @ torch.diag(model_checker[prefix + '.S']) @ model_checker[prefix + '.V']
        else:
            param.data = model_checker[name]
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean = batchnorm_state_checker[name]['m'] 
            module.running_var  = batchnorm_state_checker[name]['v'] 
    return accs, best_ranks

def to_low_rank(model, args, rank_threshold=2):
    accs = []
    ratio = args.rank_ratio

    # reinit model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model.lower() == 'resnet18':
        model_USV = ResNet18_USV_image(num_classes=1000).to(device)

    model_USV_dict = {} 
    for name, param in model_USV.named_parameters():
        model_USV_dict[name] = param.data

    # SVD
    UC, SC, SCr, VC = [], [], [], []
    UL, SL, SLr, VL = [], [], [], []

    # decomposition
    for name, param in model.named_parameters():
        # conv weight
        if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
            orig_shape = param.data.shape
            U, S, V = torch.linalg.svd(param.data.view(param.data.shape[0],-1), full_matrices=False)
            UC.append(U)
            SC.append(S)
            VC.append(V)
            SCr.append(S/torch.max(S))

        elif len(param.data.shape) > 1:
            U, S, V = torch.linalg.svd(param.data, full_matrices=False)
            UL.append(U)
            SL.append(S)
            VL.append(V)
            SLr.append(S/torch.max(S))

    SLr = torch.cat(SL)
    sorted_SLr_index = torch.argsort(SLr)

    SCr = torch.cat(SCr)
    torch.save(SCr, "SCr.pt")
    sorted_SCr_index = torch.argsort(SCr)

    # ratio
    lenL = len(sorted_SLr_index)
    lenC = len(sorted_SCr_index)

    biasL, biasC, layerL, layerC = 0, 0, 0, 0
    original_size = 0.0
    compress_size = 0.0
    ranks = []

    for name, param in model.named_parameters():
        orig_shape = param.data.shape

        original_size += param.data.numel()
        if ('conv' in name and 'weight' in name) or ('shortcut' in name and len(param.data.shape) == 4):
            num_S = len(SC[layerC])
            mask_element = torch.logical_and((sorted_SCr_index[:int(lenC*ratio)] - biasC >= 0),
                                            (sorted_SCr_index[:int(lenC*ratio)] < biasC + num_S))
            mask_element = sorted_SCr_index[:int(lenC*ratio)][mask_element] - biasC
            biasC += num_S

            S_copy = SC[layerC]
            S_copy[mask_element] = 0
            if num_S - len(mask_element) <= rank_threshold:
                S_copy[:rank_threshold] = SC[layerC][:rank_threshold]

            param.data = (UC[layerC] @ torch.diag(S_copy) @ VC[layerC]).view(orig_shape)

            prefix = '.'.join(name.split('.')[:-1])
            model_USV_dict[prefix + '.U'] = UC[layerC][:,S_copy>0]
            model_USV_dict[prefix + '.V'] = VC[layerC][S_copy>0,:]
            model_USV_dict[prefix + '.S'] = S_copy[S_copy>0]

            layerC += 1

            compress_size += model_USV_dict[prefix + '.U'].numel() + model_USV_dict[prefix + '.V'].numel()
            ranks.append(model_USV_dict[prefix + '.S'].numel())
        else:
            model_USV_dict[name] = param.data
            compress_size += param.data.numel()
    
    print(ranks, compress_size/original_size)

    # setup weight
    for name, param in model_USV.named_parameters():
        param.data = model_USV_dict[name]

    # setup batchnorm states
    batchnorm_state = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            batchnorm_state[name] = {}
            batchnorm_state[name]['m'] = module.running_mean  
            batchnorm_state[name]['v'] = module.running_var

    for name, module in model_USV.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean = batchnorm_state[name]['m'] 
            module.running_var = batchnorm_state[name]['v'] 

    return model_USV