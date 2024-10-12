import copy
import numpy as np
import os, csv, argparse
import matplotlib.pylab as plt
from torch.utils.data import DataLoader

import torch
from arch.vgg import *
from arch.resnet import *
from utility.test_model import get_test_acc
from utility.utils import compare_settings_vit, compare_settings_cnn
from utility.compress import uniform_decompose_cnn, layer_decompose_cnn, layer_decompose_cnn_iterative, layer_decompose_cnn_iterative_layer, layer_decompose_cnn_training, layer_decompose_cnn_training2target

def load_model_for_compress_cnn(args, best_setting):
    #from main import transformer
    #########################################################################
    # load the best weight
    #########################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10
    img_size = 32
    if args.dataset.lower() == 'cifar100':
        num_classes = 100

    if args.model.lower() == 'resnet18':
        model = ResNet18(num_classes=num_classes, factor=args.factor).to(device)
    elif args.model.lower() == 'resnet20':
        model = ResNet20(num_classes=num_classes, factor=args.factor).to(device)
    elif args.model.lower() == 'vgg13':
        model = VGG('VGG13',num_classes=num_classes, factor=args.factor).to(device) 
    elif args.model.lower() == 'vgg16':
        model = VGG('VGG16',num_classes=num_classes, factor=args.factor).to(device) 
  
    model.load_state_dict(torch.load(args.folder+'/'+best_setting, map_location=device)['model_state_dict'])
    if 'factor1' in best_setting:
        reference_acc = get_test_acc(model, args)
        return model, reference_acc

    #################################################################################
    # match weights for compression
    #################################################################################
    batchnorm_state = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            batchnorm_state[name] = {}
            batchnorm_state[name]['m'] = module.running_mean  
            batchnorm_state[name]['v'] = module.running_var

    model_dict = {}
    for name, param in model.named_parameters():
        # conv layers
        if 'compress' in name and '.weight' in name and 'conv' in name:
            if 'vgg' in args.model:
                layer = ''.join(name.split(":")[0])
            else:
                layer = '.'.join(name.split('.')[:-2])

            layer += '.weight'
            if layer not in model_dict:
                model_dict[layer] = param.data
            else:
                orig_shape = model_dict[layer].shape
                d = param.data.squeeze() # out, in
                model_dict[layer] = torch.matmul(d, model_dict[layer].view(model_dict[layer].shape[0],-1)).view(orig_shape)

        # fully connected layers
        elif 'compress' in name and '.weight' in name: 
            layer = '.'.join(name.split('.')[:-2])
            layer += '.weight'
            if layer not in model_dict:
                model_dict[layer] = param.data
            else:
                model_dict[layer] = torch.matmul(param.data, model_dict[layer])
        
        elif 'compress' in name and '.bias' in name:
            layer = '.'.join(name.split('.')[:-2])
            layer += '.bias'
            model_dict[layer] = param.data

        else:

            # shortcut for resnet
            if 'res' in args.model and 'shortcut' in name:
                shortcut_factor = int(name.split('.')[-2])
                if shortcut_factor == 0: # the first conv, save directly
                    model_dict[name] = param.data
                
                elif shortcut_factor == args.factor: # the last is batch norm, save directly
                    name = ".".join(name.split('.')[:-2]+["1"]+name.split('.')[-1:])
                    model_dict[name] = param.data

                else: # conv layers, combine together
                    # original layer
                    layer = ".".join(name.split('.')[:-2]+["0"]+name.split('.')[-1:])
                    orig_shape = model_dict[layer].shape
                    d = param.data.squeeze() # out, in
                    model_dict[layer] = torch.matmul(d, 
                        model_dict[layer].view(model_dict[layer].shape[0],-1)).view(d.shape[0],d.shape[1],1,1)

            else:
                model_dict[name] = param.data

    if args.model.lower() == 'resnet18':
        model = ResNet18(num_classes=num_classes, factor=1).to(device)
    elif args.model.lower() == 'resnet20':
        model = ResNet20(num_classes=num_classes, factor=1).to(device)
    elif args.model.lower() == 'vgg13':
        model = VGG('VGG13',num_classes=num_classes, factor=1).to(device) 
    elif args.model.lower() == 'vgg16':
        model = VGG('VGG16',num_classes=num_classes, factor=1).to(device) 

    for name, param in model.named_parameters():
        param.data = model_dict[name]

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if 'shortcut' in name:
                layer = ".".join(name.split('.')[:-2]+["shortcut."+str(args.factor)])
                module.running_mean = batchnorm_state[layer]['m'] 
                module.running_var = batchnorm_state[layer]['v'] 
            else:
                module.running_mean = batchnorm_state[name]['m'] 
                module.running_var = batchnorm_state[name]['v'] 

    reference_acc = get_test_acc(model, args)
    return model, reference_acc

def main(args):
    args.reference_acc = 0.0
    best_setting = compare_settings_cnn(args, dataset=args.dataset, folder=args.folder)
    model, reference_acc = load_model_for_compress_cnn(args, best_setting)
    #torch.save({'model_state_dict': model.state_dict()}, args.model+"_compress.pt")
    
    args.reference_acc = reference_acc
    if args.model == 'vgg16':
        args.reference_acc = 0.94
    if args.model == 'resnet20':
        args.reference_acc = 0.92

    # uniform decompose
    if args.global_search == 'local':
        accs = uniform_decompose_cnn(model, args)
    elif args.global_search == 'global':   
        accs, model_compress = layer_decompose_cnn(model, args)
    elif args.global_search == 'iter':
        accs, model_compress, compression_checker, best_rank = layer_decompose_cnn_iterative(model, args, grid = 2)
    elif args.global_search == 'iter_fine':
        accs, model, best_rank = layer_decompose_cnn_iterative(model, args, 
                                  grid=2, starting_rank=1)
        accs, model_compress, _ = layer_decompose_cnn_iterative(model, args, 
                                    grid=0.01, starting_rank=best_rank)
    elif args.global_search == 'iter_training':
        if args.model == 'vgg16':
            grid = 32
        elif args.model == 'resnet20':
            grid = 8
        best_ranks = []
        while grid >= 1:
            accs, best_ranks = layer_decompose_cnn_training(model, args, 
                                    grid = grid, best_ranks = best_ranks)
            print("best ranks: ", best_ranks, grid)
            grid = grid // 2

    save_path = args.save_folder+'/'+args.dataset+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = "model"+str(args.model)+"factor"+str(args.factor)+"g"+args.global_search+"ft"+str(args.finetune)+"cf"+str(args.compress_fc)+".npy"
    np.save(save_path+file_name, accs)

    try:
        torch.save({'model_state_dict': model_compress.state_dict()}, args.save_folder+'/'+best_setting)
    except:
        pass 

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='mnist', type=str, help="dataset")
    parser.add_argument("--folder", default='test', type=str, help="log folder")
    parser.add_argument("--model", default='vgg13', type=str, help="deep model")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size while training")
    parser.add_argument("--no_aug", action='store_true', help="no augmentation")
    parser.add_argument("--factor", default=2, type=int, help="how many weights are used for factorization")
    parser.add_argument("--save_folder", type=str, default='compress_cnn', help="save folder for results")
    parser.add_argument("--global_search", "-g", type=str, default="iter", help="if using global low rank")
    parser.add_argument("--compress_fc", "-cf", action='store_true', help="if compress fc layers")
    parser.add_argument("--finetune", action="store_true", help="if finetune after compression if acc drops (for global compression of CNNs only)")
    parser.add_argument("--finetune_ratio", type=float, default=1.0, help="ratio of dataset for finetuning")
    args = parser.parse_args()

    main(args)