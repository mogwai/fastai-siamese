from fastai.basics import ItemBase, ItemList, LabelList, LabelLists, CategoryList, Dataset
from copy import deepcopy
import numpy as np
import torch


class ItemTuple(ItemBase):

    def __init__(self, items):
        # Warn if the items gets larger than intended
        self.size = len(items)
        self.items = items
        self.data = torch.cat([x.data.unsqueeze(0) for x in items])

    def show(self):
        print(
            f"ItemTuple<{self.items[0].__class__.__name__}>[{len(self.items)}]")
        [x.show() for x in self.items]

    def apply_tfms(self, tfms):
        for tfm in tfms:
            self.data = torch.stack([tfm(x) for x in self.data])
        return self

    def __repr__(self):
        return ''.join([str(x)+'\n' for x in self.items])

    def __len__(self):
        return self.size



def _make_ll(ll: LabelList, pct_same=.5, tar_num=None, **kwargs):
    x = ll.x
    y = ll.y
    seperated = [x.items[y.items == i] for i in range(ll.c)]
    pairs = gen_pairs(seperated, pct_same, tar_num)
    # Keeping in sorted form
    ll.x.items = np.concatenate(seperated)
    ll.y.items = np.concatenate(
        [y.items[y.items == i] for i in range(ll.c)])
    ret = SiameseDataset(ll.x, ll.y, **kwargs)
    ret.pairs = pairs
    ret.lens = [len(i) for i in seperated]
    return ret

class SiameseDataset(LabelList):
    
    @classmethod
    def create_from_ll(cls, ll: LabelList, pct_same=0.5, tar_num=None, split_c=None, split_pct=.2, **kwargs):
        if tar_num is None:
            tar_num = len(ll.x.items)

        if isinstance(split_c, float):
            split_c = list(range(int(ll.c*split_c)))
        
        mask = None
        for i in split_c:
            if mask is None:
                mask = ll.y.items == i
            else:
                mask = np.logical_or(mask, ll.y.items == i)
        tll, vll = deepcopy(ll), deepcopy(ll)
        tll.x.items = tll.x.items[~mask]
        tll.y.items = tll.y.items[~mask]
        vll.x.items = vll.x.items[mask]
        vll.y.items = vll.y.items[mask]
        train = _make_ll(tll, pct_same=pct_same, tar_num=tar_num*(1-split_pct), **kwargs)
        valid = _make_ll(vll, pct_same=pct_same, tar_num=tar_num*split_pct, **kwargs)
        return LabelLists(ll.path, train, valid)

    def _same(self, items):
        prev = None
        for i in items:
            c = self._get_class(i)
            if prev is not None and prev != c:
                return False
            prev = c
        return True

    def _get_class(self, idx):
        total = 0
        for i, l in enumerate(self.lens):
            total += l
            if idx < total+l:
                return i

    def __len__(self):
        return len(self.pairs)

    def totuple(self, i):
        x = self.pairs[idxs]
        items = [self.x[i] for i in x]
        y = 0 if self._same(x) else 1
        x = ItemTuple(items)
        return x, y

    def __getitem__(self, idxs):
        if not isinstance(idxs, int):
            print("HELP!")
        x = self.pairs[idxs]
        items = [self.x[i] for i in x]
        y = 0 if self._same(x) else 1
        x = ItemTuple(items)
        return x, y


def gen_pairs(classes_arr, pct_same, total_pairs):
    """
    Creates a matrix of possible unique pairs and applies probabilities
    based on target total of items required.
    TODO Guaruntee one combination of each class
    TODO Add custom pairs to generated pairs
    """
    Sn = sum([len(I) for I in classes_arr])
    add_chance = 2*total_pairs / (Sn*(Sn+1)/2)
    s = Sn, Sn
    print(s)
    M = torch.rand(s).tril.t() < (1-pct_same) * add_chance
    i = 0
    for I in classes_arr:
        l = len(I)
        R[i: i+l, i: i+l] = torch.rand() < pct_same * add_chance
        i += l

    return (M*R).nonzero()
