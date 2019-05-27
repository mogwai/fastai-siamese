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
        print(str(self))
        [x.show() for x in self.items]

    def apply_tfms(self, tfms):
        for tfm in tfms:
            self.data = torch.stack([tfm(x) for x in self.data])
        return self

    def __str__(self):
        return f"ItemTuple<{self.items[0].__class__.__name__}>[{len(self.items)}]"

    def __repr__(self):
        return ''.join([str(x)+'\n' for x in self.items])

    def __len__(self):
        return self.size


def _make_ll(ll: LabelList, pct_same=.5, tar_num=None, **kwargs):
    x = ll.x
    y = ll.y
    seperated = [x.items[y.items == i] for i in np.unique(y.items)]
    l = len(x.items)
    total = (l*(l+1))/2
    pairs = compute_pairs(seperated, seperated, pct_same, tar_num/total)
    ll.x.items = np.concatenate(seperated)
    ll.y.items = np.concatenate([y.items[y.items == i] for i in np.unique(y.items)])
    ret = SiameseDataset(ll.x, ll.y, **kwargs)
    ret.pairs = pairs
    ret.lens = [len(i) for i in seperated]
    return ret


class SiameseDataset(LabelList):

    @classmethod
    def create_from_ll(cls, ll: LabelList, pct_same=0.5, tar_num=None, split_c=.2, split_pct=.2, **kwargs):
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
        train = _make_ll(tll, pct_same=pct_same,
                         tar_num=tar_num*(1-split_pct), **kwargs)
        valid = _make_ll(vll, pct_same=pct_same,
                         tar_num=tar_num*split_pct, **kwargs)
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
            # print(i,l)
            total += l
            if idx < total+l:
                return i

    def __len__(self):
        return len(self.pairs)

    def totuple(self, i):
        x = self.pairs[i]
        items = [self.x[j] for j in x]
        y = 0 if self._same(x) else 1
        x = ItemTuple(items)
        return x, y

    def __getitem__(self, idxs):
        if not isinstance(idxs, int):
            raise Exception("HELP!")
        return self.totuple(idxs)

# Total target is the percentage of getting any of the items num of items desired / total items
# pct_same is the number of same pairs to be generated implying
# the number of different pairs to be 1 - pct_same


def compute_pairs(A, B, perc=None, total_target=None, pct_same=.2):
    try:
        if not isinstance(A[0], (np.ndarray, list)):
            return gen_pairs(A, B, perc)
    except Exception as e:
        return gen_pairs(A, B, perc)

    pairs = torch.empty(0, 2, dtype=torch.long)
    right = 0

    for i in range(len(A)):
        down = 0
        for j in range(i+1):
            if total_target is not None:
                perc = pct_same if i == j else 1-pct_same
                perc *= total_target
            shift = torch.tensor([right, down])
            n = compute_pairs(A[i], B[j], perc, None) + shift
            pairs = torch.cat((pairs, n))
            down += len(B[j])
        right += len(A[i])
        
    return pairs


def gen_pairs(A, B, perc):
    # Magic number her to bump up the number to the target
    M = torch.rand(len(A), len(B)) < 3*perc
    return M.tril().t().nonzero()
