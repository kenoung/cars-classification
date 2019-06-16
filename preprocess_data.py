"""Downloads and preprocesses the `Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

We use the dataset files from the fastai mirror, and obtain the annotations from the main source directly due to
missing information in the image annotation file from the mirror.

We should obtain the following file structure after running this script.

../
    stanford-cars/
        cars_test/    - raw testing images
            ...
        cars_train/   - raw training images
            ...
        cars_test_p/  - processed testing images
            ...
        cars_train_p/ - processed testing images
            ...
    devkit/
        cars_meta.mat
        cars_test_annos.mat
        cars_train_annos.mat
        ...
"""
import logging
import os
import multiprocessing as mp
from functools import partial

import PIL
import scipy.io
from fastai.datasets import untar_data, URLs

logging.basicConfig(level=logging.INFO)
devkit_url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit'


def download_files():
    """Downloads dataset and annotations and returns the respective paths"""
    path = untar_data(URLs.CARS)
    path_devkit = untar_data(devkit_url)
    path_devkit = path_devkit.parent/'devkit'
    return path, path_devkit


def crop_by_bbox(row, folder_path, dest):
    xmin, xmax, ymin, ymax, fname = row[0][0][0], row[1][0][0], row[2][0][0], row[3][0][0], row[-1][0]
    bbox = (xmin, xmax, ymin, ymax)
    PIL.Image.open(folder_path / fname).crop(bbox).save(dest / fname)


def preprocess(folder_path, annotations, dest, use_multi=True):
    """Preprocess data by cropping out the relevant area of interest using provided annotations"""
    if use_multi:
        with mp.Pool() as p:
            logging.info("using {} processes".format(p._processes))
            p.map(partial(crop_by_bbox, folder_path=folder_path, dest=dest), annotations)
    else:
        for row in annotations:
            crop_by_bbox(row, folder_path, dest)


def process(path, path_devkit):
    path_test = path/'cars_test'
    path_test_p = path/'cars_test_p'
    path_train = path/'cars_train'
    path_train_p = path/'cars_train_p'
    path_test_annos = path_devkit/'cars_test_annos.mat'
    path_train_annos = path_devkit/'cars_train_annos.mat'

    os.makedirs(path_train_p, exist_ok=True)
    os.makedirs(path_test_p, exist_ok=True)

    train_annos = scipy.io.loadmat(path_train_annos)['annotations'][0]
    test_annos = scipy.io.loadmat(path_test_annos)['annotations'][0]
    preprocess(path_train, train_annos, path_train_p)
    preprocess(path_test, test_annos, path_test_p)


if __name__ == '__main__':
    path, path_devkit = download_files()
    logging.info("image_dir: {}, annos_dir: {}".format(path, path_devkit))

    process(path, path_devkit)

