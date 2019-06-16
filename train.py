from fastai.callbacks import SaveModelCallback
from fastai.metrics import error_rate
from fastai.vision import cnn_learner, models

from util import get_data


def train(arch):
    data = get_data('train')
    learn = cnn_learner(data, arch, metrics=error_rate).mixup()
    learn.fit_one_cycle(
        15,
        callbacks=[SaveModelCallback(learn, monitor='error_rate', mode='min', name='bestmodel')]
    )

    learn.load('bestmodel')
    learn.unfreeze()
    learn.fit_one_cycle(
        5, max_lr=slice(1e-6, 1e-4),
        callbacks=[SaveModelCallback(learn, monitor='error_rate', mode='min', name='bestmodel-unfreeze')])

if __name__ == '__main__':
    arch = models.resnet18
    train(arch)

