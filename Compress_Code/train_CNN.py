import sys; sys.path.append("..")

import numpy as np
import argparse, os, time
import torch, torchvision

from torch.optim import SGD
from data.cifar import Cifar10, Cifar100, TinyImageNet
from utility.log import Log
from arch.resnet import *
from arch.vgg import *
from utility.initialize import initialize
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='vgg13', type=str, help="select model")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=2000, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", '-lr', default=1e-1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", '-beta', default=0.6, type=float, help="SGD Momentum.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset name")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0000, type=float, help="L2 weight decay.")
    parser.add_argument("--seed", default=42, type=int, help="L2 weight decay.")
    parser.add_argument("--patience", default=300, type=int, help="patience for scheduler.")
    parser.add_argument("--scheduler", default='stepLR', type=str, help="select scheduler.")
    parser.add_argument("--multigpu", "-m", action='store_true', help="whether using multi-gpus.")
    parser.add_argument("--optimizer", type=str, default='Adam', help="SGD/Adam")
    parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout")
    parser.add_argument("--augmentation", "-aug", action='store_true', help="if using augmentation.")
    #parser.add_argument("--logs", default='cnns', action='store_true', help="if resume training")
    parser.add_argument("--result_folder", default='base', type=str, help="result_folder.")
    parser.add_argument("--factor", default=1, type=int, help="factorization.")
    parser.add_argument("--resume", default="", type=str, help="if loading weights")
    args = parser.parse_args()

    initialize(args, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = 10
    size = (32, 32)
    if args.dataset.lower() == 'cifar10':
        dataset = Cifar10(args.batch_size, args.threads, size, args.augmentation)
    elif args.dataset.lower() == 'cifar100':
        dataset = Cifar100(args.batch_size, args.threads, size, args.augmentation)
        labels = 100
    elif args.dataset.lower() == 'fashionmnist':
        dataset = FashionMNIST(args.batch_size, args.threads, size)
    elif args.dataset.lower() == 'tiny':
        size = (64, 64)
        dataset = TinyImageNet(args.batch_size, args.threads, size)
    else:
        dataset = MNIST(args.batch_size, args.threads, size)
        
    if args.model.lower() == 'resnet18':
        model = ResNet18(num_classes=labels, factor=args.factor, input_res=size[0]).to(device)
    elif args.model.lower() == 'resnet20':
        model = ResNet20(num_classes=labels, factor=args.factor, input_res=size[0]).to(device)
    # elif args.model.lower() == 'resnet34':
    #     model = ResNet34(num_classes=labels).to(device)
    # elif args.model.lower() == 'resnet50':
    #     model = ResNet50(num_classes=labels).to(device)
    # elif args.model.lower() == 'densenet':
    #     model = DenseNet121(num_classes=labels).to(device)
    # elif args.model.lower() == 'wide16':
    #     model = Wide_ResNet(depth=16, widen_factor=4, 
    #         dropout_rate=args.dropout, num_classes=labels).to(device)
    #     model.apply(conv_init)
    if args.model.lower() == 'vgg13':
        model = VGG('VGG13',num_classes=labels, factor=args.factor).to(device) 
    elif args.model.lower() == 'vgg16':
        model = VGG('VGG16',num_classes=labels, factor=args.factor,  input_res=size[0]).to(device)      
    elif args.model.lower() == 'vgg16c':
        model = VGG('VGG16_Compress',num_classes=labels, factor=args.factor,  input_res=size[0]).to(device)    
    if args.multigpu:
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    
    file_name = (args.dataset+'lr'+str(args.learning_rate)
                  +'beta'+str(args.momentum)
                  +'batchsize'+str(int(args.batch_size))
                  +'model'+str(args.model)
                  +'seed'+str(args.seed)
                  +'scheduler'+args.scheduler
                  +'patience'+str(args.patience)
                  +'optimizer'+args.optimizer
                  +'wd'+str(args.weight_decay)
                  +'dp'+str(args.dropout)
                  +'lb'+str(args.label_smoothing)
                  +'aug'+str(args.augmentation)
                  +'factor'+str(args.factor)
            )

    if len(args.resume) > 0:
         model.load_state_dict(torch.load(args.resume, map_location=device)['model_state_dict'])

    args.logs = args.result_folder+'/'+args.dataset+'/'+args.model+'/'
    log = Log(log_each=10, file_name=file_name, logs='./'+args.result_folder+'/'+args.dataset+'/'+args.model+'/')
    criterion = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=args.label_smoothing)
    
    if args.optimizer.lower() == 'sgd':
       optimizer = SGD(model.parameters(),lr=args.learning_rate, momentum=args.momentum, 
                       weight_decay=args.weight_decay, nesterov=False)
    elif args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=args.weight_decay)
        
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)#originally 20
    achieve_target_acc = 0
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        for batch in dataset.train:
            start_time = time.time()

            inputs, targets = (b.to(device) for b in batch)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)

            start_time2 = time.time()
            loss.mean().backward()

            optimizer.step()

            correct = torch.argmax(predictions.data, 1) == targets
            log(model, loss.cpu(), correct.cpu(), optimizer.param_groups[0]['lr'])
            #print("time:", time.time() - start_time)
                 
        train_acc = log.epoch_state["accuracy"] / log.epoch_state["steps"]
        if train_acc >= 0.999:
            achieve_target_acc += 1
            if achieve_target_acc > 20:
                break

        # no need to keep training
        if optimizer.param_groups[0]['lr'] < 1e-5:
            break

        model.eval()
        log.eval(len_dataset=len(dataset.test))
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu(), optimizer.param_groups[0]['lr'])
        
        scheduler.step(train_acc)
        test_acc = log.epoch_state["accuracy"] / log.epoch_state["steps"]
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict()}, args.logs+"/"+file_name+".pt")

    log.flush()
