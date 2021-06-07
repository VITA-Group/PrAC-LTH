import copy 
import torch
import numpy as np 

from dataset import *

from models.vgg import vgg16_bn
from models.resnet import resnet18, resnet50
from models.resnets import resnet20, resnet56

from advertorch.utils import NormalizeByChannelMeanStd



__all__ = ['setup_model_dataset', 'setup_model_dataset_PIE',
            '']



def setup_model_dataset(args, if_train_set=False):

    if args.dataset == 'cifar10':
        classes = 10
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, dataset = if_train_set)

    elif args.dataset == 'cifar100':
        classes = 100
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, dataset = if_train_set)

    elif args.dataset == 'tiny-imagenet':
        classes = 200
        train_number = 90000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_set_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data, dataset = if_train_set, split_file = args.split_file)
    else:
        raise ValueError('unknow dataset')

    if args.arch == 'res18':
        print('build model resnet18')
        model = resnet18(num_classes=classes, imagenet=True)
    elif args.arch == 'res50':
        print('build model resnet50')
        model = resnet50(num_classes=classes, imagenet=True)
    elif args.arch == 'res20s':
        print('build model: resnet20')
        model = resnet20(number_class=classes)
    elif args.arch == 'res56s':
        print('build model: resnet56')
        model = resnet56(number_class=classes)
    elif args.arch == 'vgg16_bn':
        print('build model: vgg16_bn')
        model = vgg16_bn(num_classes=classes)
    else:
        raise ValueError('unknow model')

    model.normalize = normalization

    if if_train_set:
        return model, train_set_loader, val_loader, test_loader, train_number
    else:
        return model, train_set_loader, val_loader, test_loader


def setup_model_dataset_PIE(args):

    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader = cifar10_dataloaders_val(batch_size = args.batch_size, data_dir = args.data)

    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_loader = cifar100_dataloaders_val(batch_size = args.batch_size, data_dir = args.data)

    elif args.dataset == 'tiny-imagenet':
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader = tiny_imagenet_dataloaders_val(batch_size = args.batch_size, data_dir = args.data, split_file=args.split_file)

    else:
        raise ValueError('unknow dataset')

    if args.arch == 'res18':
        print('build model resnet18')
        model = resnet18(num_classes=classes, imagenet=True)
    elif args.arch == 'res50':
        print('build model resnet50')
        model = resnet50(num_classes=classes, imagenet=True)
    elif args.arch == 'res20s':
        print('build model: resnet20')
        model = resnet20(number_class=classes)
    elif args.arch == 'res56s':
        print('build model: resnet56')
        model = resnet56(number_class=classes)
    elif args.arch == 'vgg16_bn':
        print('build model: vgg16_bn')
        model = vgg16_bn(num_classes=classes)
    else:
        raise ValueError('unknow model')

    model.normalize = normalization

    return model, train_loader




# other function
def forget_times(record_list):
    
    offset = 200000
    number = offset
    learned = False

    for i in range(record_list.shape[0]):
        
        if not learned:
            if record_list[i] == 1:
                learned = True 
                if number == offset:
                    number = 0

        else:
            if record_list[i] == 0:
                learned = False
                number+=1 

    return number

def sorted_examples(example_wise_prediction, data_prune, data_rate, state, threshold, train_number):

    offset = 200000

    forgetting_events_number = np.zeros(example_wise_prediction.shape[0])
    for j in range(example_wise_prediction.shape[0]):
        tmp_data = example_wise_prediction[j,:]
        if tmp_data[0] < 0:
            forgetting_events_number[j] = -1 
        else:
            forgetting_events_number[j] = forget_times(tmp_data)

    # print('* never learned image number = {}'.format(np.where(forgetting_events_number==offset)[0].shape[0]))

    if data_prune == 'constent':
        print('* pruning {} data'.format(data_rate))
        rest_number = int(train_number*(1-data_rate)**state)
    elif data_prune == 'zero_out':
        print('zero all unforgettable images out')
        rest_number = np.where(forgetting_events_number > threshold)[0].shape[0]
    else:
        print('error data_prune type')
        assert False

    # print('max forgetting times = {}'.format(np.max(forgetting_events_number)))
    selected_index = np.argsort(forgetting_events_number)[-rest_number:]

    return selected_index

def split_class_sequence(sequence, all_labels, num_class):
    
    class_wise_sequence = {}
    for i in range(num_class):
        class_wise_sequence[i] = []
    
    for index in range(sequence.shape[0]):
        class_wise_sequence[all_labels[sequence[index]]].append(sequence[index])
    
    for i in range(num_class):
        class_wise_sequence[i] = np.array(class_wise_sequence[i])
        print('class = {0}, number = {1}'.format(i, class_wise_sequence[i].shape[0]))

    return class_wise_sequence

def blance_dataset_sequence(class_wise_sequence, num_class):

    class_wise_number = np.zeros(num_class, dtype=np.int)
    for i in range(num_class):
        class_wise_number[i] = class_wise_sequence[i].shape[0]
    
    max_length = np.max(class_wise_number)
    print('max class number = {}'.format(max_length))

    balance_sequence = []
    arange_max = np.arange(max_length)
    for i in range(num_class):

        shuffle_index = np.random.permutation(class_wise_number[i])
        shuffle_class_sequence = class_wise_sequence[i][shuffle_index]
        balance_sequence.append(shuffle_class_sequence[arange_max%class_wise_number[i]])

    balance_sequence = np.concatenate(balance_sequence)
    print(balance_sequence.shape)
    return balance_sequence



