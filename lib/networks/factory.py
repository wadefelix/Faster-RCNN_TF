# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}


def get_network(name):
    """Get a network by name."""

    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
            from networks.VGGnet_test import VGGnet_test as VGGnet_test
            return VGGnet_test()
        elif name.split('_')[1] == 'train':
            from networks.VGGnet_train import VGGnet_train as VGGnet_train
            return VGGnet_train()
        else:
           raise KeyError('Unknown network: {}'.format(name))
    elif name.split('_')[0] == 'resnet':
        if name.split('_')[1] == 'test':
            from networks.resnet_test import resnet_test as resnet_test
            return resnet_test()
        elif name.split('_')[1] == 'train':
            from networks.resnet_train import resnet_train as resnet_train
            return resnet_train()
        else:
            raise KeyError('Unknown network: {}'.format(name))
    elif name.split('_')[0] == 'Resnet50':
        if name.split('_')[1] == 'test':
            from networks.Resnet50_test import Resnet50_test as Resnet50_test
            return Resnet50_test()
        elif name.split('_')[1] == 'train':
            from networks.Resnet50_train import Resnet50_train as Resnet50_train
            return Resnet50_train()
        else:
            raise KeyError('Unknown network: {}'.format(name))
    else:
        raise KeyError('Unknown network: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
