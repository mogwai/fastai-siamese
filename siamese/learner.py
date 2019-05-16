from fastai.basics import *
from fastai.vision import models
from torch import nn
from copy import deepcopy

class SiameseNetwork(nn.Module):

    def __init__(self, encoder=models.resnet18, s_out=512):
        # TODO warn is s_out is to large
        super().__init__()
        self.body = create_body(encoder, cut=-2)
        self.head = create_head(1024, 1, [s_out])[:5]

    def forward(self, items):
        # Get the batch size in the correct place
        ins = items.transpose(1, 0)
        outs = [self.body(x) for x in ins]
        outs = [self.head(x) for x in outs]
        outs = torch.stack(outs)
        return outs


def hinge_loss(x, y, m=1):
    # Was getting autograd errors if I didn't clone
    diff = torch.sqrt(torch.pow(x[0]-x[1], 2)).clone()**2
    diff[y == 0] = m**2 - diff[y == 0]
    diff[diff < 0] = 0
    return diff


def gen_loss_m(loss_func):
    return lambda x, y: loss_func(x, y).mean()


def loss_acc(loss_func, l):
    return lambda x, y: (loss_func(x, y) < l**2).float().mean()


def create_loss_acc(loss_func, l):
    return gen_loss_m(loss_func), loss_acc(loss_func, l)


def siamese_learner(data: DataBunch, encoder: nn.Module = models.resnet18, s_out=512, loss_func=None, loss_size=None, m=3):
    if loss_func is None:
        loss_func = partial(hinge_loss, m=m)
    # m/2 is the middle of confidence so if we're bellow it that means we've guessed correctly
    # although with very low confidence
    if loss_size is None:
        loss_size = m/2
    loss, acc = create_loss_acc(loss_func, loss_size)
    learner = Learner(data, SiameseNetwork(encoder, s_out),
                      loss_func=loss, metrics=acc)
    # TODO create a simple way to get a vector of a piece of data
    learner.encode = lambda x: x
    return learner
