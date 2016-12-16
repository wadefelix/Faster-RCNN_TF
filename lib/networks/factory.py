# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import pdb
import tensorflow as tf

#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()


def get_network(name):
    """Get a network by name."""
    #if not __sets.has_key(name):
    #    raise KeyError('Unknown dataset: {}'.format(name))
    #return __sets[name]
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
            import networks.VGGnet_test
            return networks.VGGnet_test()
        elif name.split('_')[1] == 'train':
            import networks.VGGnet_train
            return networks.VGGnet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    elif name.split('_')[0] == 'resnet':
        if name.split('_')[1] == 'test':
            import networks.resnet_test
            return networks.resnet_test()
        elif name.split('_')[1] == 'train':
            import networks.resnet_train
            return networks.resnet_train.resnet_train()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
