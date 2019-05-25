from fastai.basics import ItemBase, ItemList, LabelList, CategoryList, Dataset
from copy import deepcopy
import numpy as np


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
        return ''.join([str(x) for x in self.items]) + '\n'

    def __len__(self):
        return self.size


class SiameseList(ItemList):

    def __init__(self, *args, open_fn=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.open = open_fn

    def get(self, i):

        # i is an integer then we convert to 2d
        item = super().get(i)
        if isinstance(item, ItemTuple):
            return item
        
        return ItemTuple([self.open(x) for x in item])


class SiameseDataset(LabelList):

    @classmethod
    def create_from_ll(cls, ll: LabelList, bs: int = 1, pct_same=0.5):
        x = ll.x
        y = ll.y
        seperated = [x.items[y == i] for i in range(ll.c)]

    def __len__(self);
        return len(self.x,items)


"""
Questions
- How do we make sure there is a percentage of similar pairs?
- Ensure that everything is paired up atleast once?
- Is there a mathemtical way to do this?


- Specify the number of total pairs
- Percentage the same, (implying percentage different)
- 

1. Get given a labelled list of items with paramaters
2. Generate from that add the indexs of the pairs as if the items had been stacked up against each other
    a1 , ... , an , b1 , ... , bm ,
3. 
"""
