#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import re

import datetime
import Tkinter as tk
import tkFont
import tkFileDialog

from gapt import GaptTask
import pprint
import config as mcfg

from networks.factory import get_network
import tensorflow as tf


CLASSES = mcfg.cfg.CLASSES
EDGECOLOR = mcfg.cfg.EDGECOLOR

# use matplot first, then opencv or some other replace the matplot
def plot_detections_on_image(im, dets, savename=None):
    """
    dets : [bbox,score, cls_ind]
    class_name = CLASSES[cls_ind].
    """
    inds = np.where(dets[:, 5] > 0.1)[0]
    if len(inds) == 0:
        if (savename is not None): cv2.imwrite(savename, im)
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -2]
        classname = CLASSES[int(dets[i, -1])]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=EDGECOLOR[classname], linewidth=3.5)
            )
        
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(classname, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    # save and close
    if (savename is not None):
        plt.savefig(savename)
    plt.close()

def detection(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    CONF_THRESH = mcfg.cfg.CONF_THRESH
    NMS_THRESH = mcfg.cfg.NMS_THRESH
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    resdets = np.zeros((100,6));
    resdets_rowptr = 0;
    for cls_ind, cls in enumerate(CLASSES[1:]):
        if len(EDGECOLOR[cls])==0: continue
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in range(len(inds)):
            box = dets[inds[i],:]
            # print box
            box_h = box[3]-box[1]
            box_w = box[2]-box[0]
            if box_h<=0 or box_w <=0 or (abs(box[2] - im.shape[1])<3 and (box_h/box_w>2) and box_w < 35) or (box_h/box_w>4):
                inds[i] = -1
        inds = inds[np.where(inds>=0)]
        if len(inds) == 0:
            continue
        else:
            if resdets_rowptr + len(inds) < resdets.shape[0]:
                resdets_inds = np.arange(len(inds)) + resdets_rowptr
                resdets[resdets_inds,0:5] = dets[inds,:]
                resdets[resdets_inds,5:6] = np.ones((len(inds),1)) * cls_ind
                resdets_rowptr += len(inds)

    return resdets

def det2result(dets):
    inds = np.where(dets[:, 5] > 0.1)[0]
    if len(inds) == 0:
        return ''
    result = []
    for i in inds:
        bbox = dets[i, :4]
        #score = dets[i, -2]
        classname = CLASSES[int(dets[i, -1])]
        #if classname in ("knife",): continue
        result.append('['+ ','.join([str(int(round(x))) for x in bbox.tolist()]) + ','+str(dets[i,4]) + ','+classname +']')
    return ';'.join(result)


def runtest(args):
  # init session
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # load network
    net = get_network(args.net)
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    print '\n\nLoaded network {:s} checkpoint {:s}'.format(args.net, args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    mcfg.cfg.CONF_THRESH = float(args.CONF_THRESH)
    mcfg.cfg.NMS_THRESH = float(args.NMS_THRESH)
    
    task = GaptTask(args.gapt_svr)
    task.Create(args.gapt_temp_id, args.gapt_task_title, args.gapt_task_descrip + args.model + ', CONF_THRESH={}, NMS_THRESH={}'.format(mcfg.cfg.CONF_THRESH,mcfg.cfg.NMS_THRESH));
    while True:
        im = task.TestGetData()
        if im is None: break
        dets = detection(sess,net,im)
        task.TestSaveResult(det2result(dets))
    print "task_id = {} done. {}/task_list.php".format(task.GetTaskInfo('task_id'),args.gapt_svr)
  # clean all
  tf.reset_default_graph()


class DisplayApp(tk.Frame):
    def __init__(self, master = None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        top=self.winfo_toplevel() 
        top.rowconfigure(0, weight=1) 
        top.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        #top.protocol("WM_DELETE_WINDOW",self.close)
        
        self.screenwidth =  600 #top.winfo_screenwidth()
        self.screenheight = 400 #top.winfo_screenheight()

        self.F = {}
        self.F['param'] = tk.Frame(top)
        self.F['param'].grid(row=1,column=1)
        self.F['button'] = tk.Frame(top)
        self.F['button'].grid(row=2,column=1)

        self.textvariables = {}
        self.labels = {}

        self.entryid = 1;
        self.entries = {}
        self.makeEntry('net','VGGnet_test')
        self.makeEntry('model')
        self.makeEntry('GAPT server','http://127.0.0.1/gapt/')
        self.makeEntry('templateid')
        self.makeEntry('task title',datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f'))
        self.makeEntry('task description')
        self.makeEntry('CONF THRESH','0.1')
        self.makeEntry('NMS THRESH','0.3')
        
        self.initdir = os.path.realpath('.')
        self.OPENMODEL = tk.Button(self.F['model'])
        self.OPENMODEL["text"] = "..."
        self.OPENMODEL["command"] =  self.askopenfilename_model
        self.OPENMODEL.pack({"side": "left"})

        self.F['QUIT'] = tk.Frame(self.F['button'])
        self.F['QUIT'].grid(row=1,column=2)
        self.QUIT = tk.Button(self.F['QUIT'])
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.close

        self.QUIT.pack({"side": "left"})

        self.F['RUN'] = tk.Frame(self.F['button'])
        self.F['RUN'].grid(row=1,column=3)
        self.RUN = tk.Button(self.F['RUN'])
        self.RUN["text"] = "RUN",
        self.RUN["command"] = self.run

        self.RUN.pack({"side": "right"})
        
        top.bind("<Configure>", self.resize)
    def makeEntry(self, name, default='', cmd=None):
        self.F[name] = tk.Frame(self.F['param'])
        self.F[name].grid(row=self.entryid,column=1)

        self.labels[name] = tk.Label(self.F[name],text=name, width=20)
        self.labels[name].pack({"side": "left"})
        self.textvariables[name] = tk.StringVar()
        self.textvariables[name].set(default)
        self.entries[name] = tk.Entry(self.F[name],textvariable=self.textvariables[name] , width = 48);
        self.entries[name].pack({"side": "left"})
        self.entryid += 1
    
    def askopenfilename_model(self):
        filename = tkFileDialog.askopenfilename(initialdir = self.initdir, filetypes = [('Supported types',('.ckpt','.ckpt.*')), ('tensorflow checkpoint v1 file', '.ckpt'), ('tensorflow checkpoint v2 files', '.ckpt.*'), ('all files', '*')])
        if filename:
            res = re.search(r'.*\.ckpt',filename)
            if res:
                filename = res.group(0)
                self.textvariables['model'].set(filename)
                self.initdir = os.path.dirname(filename)
    def close(self):
        self.quit()
    def resize(self, event):
        #print "resize called :" + str(event.width) + "x" + str(event.height)
        pass
    def run(self):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        class myobj():
            pass
        args = myobj()
        args.net = self.textvariables['net'].get()
        args.model = self.textvariables['model'].get()
        args.gapt_svr = self.textvariables['GAPT server'].get()
        args.gapt_temp_id = self.textvariables['templateid'].get()
        args.gapt_task_title = self.textvariables['task title'].get()
        args.gapt_task_descrip = self.textvariables['task description'].get()
        args.CONF_THRESH = self.textvariables['CONF THRESH'].get()
        args.NMS_THRESH = self.textvariables['NMS THRESH'].get()
        runtest(args)
        print reconstructcmdline(args)

def startDisplayApp():
    disp = DisplayApp()
    disp.master.title('GAPT')
    #disp.DisplayResultHandler()
    disp.mainloop()


def reconstructcmdline(args):
    global arguments
    strargs = [sys.argv[0]]
    for item in arguments:
        if hasattr(args,item[1]):
            strargs.append(item[0])
            strargs.append('"{}"'.format(str(getattr(args,item[1]))))
    return " ".join(strargs)

arguments = [
# flags, dest, type, default, help
#['--gpu', 'gpu_id', int, 0, 'GPU id to use'],
['--net', 'net', str, 'VGGnet_test', 'Network to use [vgg16]'],
['--model', 'model', str, None, 'Model path'],
['--server', 'gapt_svr',str,'http://127.0.0.1/gapt/', 'GATP Server Address'],
['--templateid', 'gapt_temp_id', str, '22', 'GATP Template ID'],
['--tasktitle', 'gapt_task_title', str, datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f'), 'GATP Task Description'],
['--taskdescrip', 'gapt_task_descrip', str, '', 'GATP Task Description'],
['--CONF_THRESH', 'CONF_THRESH', float, 0.1, 'GATP Task Description'],
['--NMS_THRESH', 'NMS_THRESH', float, 0.3, 'GATP Task Description'],
]

def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(description='run gapt Test')
    #parser.add_argument('--cpu', dest='cpu_mode', action="store_true",
    #                    help='Use CPU mode (overrides --gpu)')
    parser.add_argument('--gui', action="store_true")
    for item in arguments:
        parser.add_argument(item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

    if len(sys.argv) == 1:
        # display gui
        args = parser.parse_args(['--gui'])
        return args
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    if args.gui:
        startDisplayApp()
        sys.exit(0)

    print('Using config:')
    pprint.pprint(args)

    CONF_THRESH = args.CONF_THRESH
    NMS_THRESH = args.NMS_THRESH

    if args.model is not None:
        runtest(args)
    else:
        pass



