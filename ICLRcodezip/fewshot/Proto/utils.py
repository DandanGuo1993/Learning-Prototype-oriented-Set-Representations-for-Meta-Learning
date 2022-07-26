import numpy as np
import os
import glob
import backbone
import torch

model_dict = dict(
            Conv4 = backbone.Conv4,
            ResNet10 = backbone.ResNet10,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50)


def get_resume_file(checkpoint_dir, is_train=False):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    if is_train:
        filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
        epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
        max_epoch = np.max(epochs)
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    else:
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(40))
    return resume_file


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)
