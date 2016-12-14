#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Copyright (c) 2016 MergeSecurity
# Written by Ren Wei
# --------------------------------------------------------

"""
ATR Algorithm Server Configuration
"""

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__CLASSESINFO = {'__background__':[0,'',''],
           'fruits':[1,'Fruits','blue'],
           'umbrella':[2,'Umbrella','red'],
           'hardnegative':[3,'',''],
           'wallet':[4,'Wallet','blue'],
           'bottle':[5,'Bottle','red'],
           'keys':[6,'Keys','blue'],
           'chargers':[7,'Chargers','blue'],
           'laptop':[8,'Laptop','blue'],
           'tablet':[9,'Tablet','blue'],
           'coins':[10,'Coins','blue'],
           'selfiestick':[11,'Selfie Stick','blue'],
           'dog':[12,'',''],
           'horse':[13,'',''],
           'motorbike':[14,'',''],
           'hdd':[15,'HDD','blue'],
           'camera':[16,'DSLR Camera','blue'],
           'knife':[17,'Blade','red'],
           'glassescase':[18,'Eyeglass Case','blue'],
           'battery':[19,'Cylindrical Cell','DarkOrange'],
           'smartphone':[20,'Mobile Phone','blue']
           }

__listCLASSES = [''] * len(__CLASSESINFO)
for key,val in __CLASSESINFO.iteritems():
    __listCLASSES[val[0]] = key

__C.CLASSES = __listCLASSES

__C.EDGECOLOR = {key:val[2] for key,val in __CLASSESINFO.iteritems()}

__C.DISPLAYNAME = {key:val[1] for key,val in __CLASSESINFO.iteritems()}

__C.CONF_THRESH = 0.89
__C.NMS_THRESH = 0.3
           
