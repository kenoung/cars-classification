import os

import torch
from pathlib import PosixPath

import scipy.io
from fastai.vision import get_transforms, imagenet_stats, ImageDataBunch


def get_data(image_set='train', path='~/.fastai/data/stanford-cars', path_devkit='~/.fastai/data/devkit',
             normalization_type='default'):
    """
    Returns an ImageDataBunch for the specified image_set

    Parameters
    ----------
    image_set: str
    One of {'train', 'test'}

    path: PosixPath or str
    Image directory path

    path_devkit: PosixPath or str
    Devkit directory path

    normalization_type: str
        One of {'default', 'imagenet'}

    Returns
    -------
    data: ImageDataBunch obj
    """
    if type(path) == str:
        path = PosixPath(os.path.expanduser(path))

    if type(path_devkit) == str:
        path_devkit = PosixPath(os.path.expanduser(path_devkit))

    path_images = path/'cars_{}_p'.format(image_set)
    path_annos = path_devkit/'cars_{}_annos.mat'.format(image_set)
    path_labels_names = path_devkit/'cars_meta.mat'

    annotations = scipy.io.loadmat(path_annos)['annotations'][0]
    label_names = scipy.io.loadmat(path_labels_names)['class_names'][0]

    images = [path_images/x[5][0] for x in annotations]
    labels = [label_names[int(x[4][0])-1][0] for x in annotations]

    if normalization_type == 'default':
        normalization_stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])]
    elif normalization_type == 'imagenet':
        normalization_stats = imagenet_stats
    else:
        raise ValueError("invalid normalization_type provided")

    data = ImageDataBunch.from_lists(path_images, fnames=images, labels=labels, ds_tfms=get_transforms(), size=331, bs=8
                                      ).normalize(normalization_stats)
    return data
