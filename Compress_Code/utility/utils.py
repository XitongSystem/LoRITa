import os, csv
import numpy as np

# plot training and testing curve
def plot_curves(file, save_path=None):        
    ##########################################################
    # load logs
    ##########################################################
    train_losses = []
    train_acces = []

    test_losses = []
    test_acces = []
    with open(file) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0][:5] != 'Epoch':
                continue
            loss = row[0].split(":")[1].split(' ')[1][:-1]
            acc = row[0].split(":")[-1]

            if 'train' in row[0]:
                train_losses.append(float(loss))
                train_acces.append(float(acc))
            else:
                test_losses.append(float(loss))
                test_acces.append(float(acc))

    ##########################################################
    # plot
    ##########################################################
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.figure()
        plt.plot(train_acces, label='train_acc')
        plt.plot(test_acces, label='test_acc')
        plt.title("Test Acc: "+str(max(test_acces))+" Train Acc: "+str(max(train_acces)))
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path+'/'+file.split('/')[-1][:-3]+'.png')
    return max(test_acces)

def compare_settings_vit(args, dataset='mnist', folder='test', base=False, tag='comp'):
    ##########################################################
    # list dir and select the best
    ##########################################################
    files = os.listdir(folder)
    best_setting = '' 
    best_test = 0
    for f in files:
        if ((base and tag not in f) or (base == False and tag in f and ('fc{'+str(args.factor)) in f)) and f[-4:]=='.out':
            #try:
            test_acc = plot_curves(folder+'/'+f)
            if best_test < test_acc:
                best_setting = f
                best_test = test_acc
            # except:
            #     pass

    print(base, best_setting, best_test)
    ##########################################################
    # transform setting to the file name 
    ##########################################################
    lr = best_setting.split('_')[1]
    lr = float(lr[lr.find("{")+1:lr.find("}")])
    wd = best_setting.split('_')[2]
    wd = float(wd[wd.find("{")+1:wd.find("}")])
    bs = "128"

    weight_name = ('data' + dataset 
                + '_hd' + str(args.head)
                + '_d' + str(args.dim)
                + '_lr' + str(lr) 
                + '_bs' + str(bs) 
                + '_wd' + str(wd) 
                + '_optadam' 
                #+ '_cp' + str(not(base))
                + '_fc' + str(args.factor)
                + '_dp' + str(args.depth)
                + '.pt'
    )
    return weight_name


def compare_settings_cnn(args, dataset='mnist', folder='test'):
    def read(logs):
        my_file = open(logs, "r")
        data = my_file.read().split('\n')
        my_file.close()
        test_acc = []
        for d in data:
            if "┃" not in d:
                continue
            line = d.split("┃")
            test_acc.append(round(float(line[-2].split("│")[1][:-3])/100.,4))
        test_acc = np.array(test_acc)
        return np.mean(test_acc[-5:])

    ##########################################################
    # list dir and select the best
    ##########################################################
    files = os.listdir(folder)
    best_setting = '' 
    best_test = 0
    for f in files:
        if f[-4:]=='.txt' and ('factor'+str(args.factor)) in f:
            #try:
            test_acc = read(folder+'/'+f)
            if best_test < test_acc:
                best_setting = f
                best_test = test_acc
            # except:
            #     pass

    print(best_setting, best_test)
    weight_name = best_setting[:-4]+'.pt'
    return weight_name


def compare_settings_fcn(args, dataset='mnist', folder='test', base=False, tag='comp'):
    ##########################################################
    # list dir and select the best
    ##########################################################
    files = os.listdir(folder)
    best_setting = '' 
    best_test = 0
    for f in files:
        if ((base and tag not in f) or (base == False and tag in f and ('fc{'+str(args.factor)) in f)) and f[-4:]=='.out':
            #try:
            test_acc = plot_curves(folder+'/'+f)
            if best_test < test_acc:
                best_setting = f
                best_test = test_acc
            # except:
            #     pass

    print(base, best_setting, best_test)
    ##########################################################
    # transform setting to the file name 
    ##########################################################
    lr = best_setting.split('_')[1]
    lr = float(lr[lr.find("{")+1:lr.find("}")])
    wd = best_setting.split('_')[2]
    wd = float(wd[wd.find("{")+1:wd.find("}")])
    bs = "128"

    weight_name = ('data' + dataset 
                + '_d' + str(args.dim)
                + '_lr' + str(lr) 
                + '_bs' + str(bs) 
                + '_wd' + str(wd) 
                + '_optadam' 
                + '_fc' + str(args.factor)
                + '_dp' + str(args.depth)
                + '.pt'
    )

    # weight_name = ('data' + dataset 
    #             + '_d' + str(args.dim)
    #             + '_lr' + str(lr) 
    #             + '_bs' + str(bs) 
    #             + '_wd' + str(wd) 
    #             + '_optsgd' 
    #             + '_fc' + str(args.factor)
    #             + '_dp' + str(args.depth)
    #             + '.pt'
    # )

    return weight_name