from fastai.basics import ItemBase, ItemList, LabelList, CategoryList
from copy import deepcopy


class ItemTuple(ItemBase):

    def __init__(self, items):
        # Warn if the items gets larger than intended
        self.size = len(items)
        self.items = items
        self.data = torch.cat([x.data.unsqueeze(0) for x in items])

    def show(self):
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
        item = super().get(i)
        if isinstance(item, ItemTuple):
            return item
        return ItemTuple([self.open(x) for x in item])


class SiameseDataset(LabelList):

    def show_similar(self):
        pass

    def show_different(self):
        pass

    @classmethod
    def from_label_list(cls, ll: LabelList, hidden_classes=None, train_num=20, valid_num=10, use_all=False):
        # Use random chunk of the classes for validation if none specified
        if hidden_classes is None:
            hidden_classes = list(range(ll.c))[-int(ll.c*0.2)]

        if hasattr(hidden_classes[0], 'data'):
            hidden_classes = [x.data for x in hidden_classes]

        if len(hidden_classes) < 2:
            raise Exception("Must be atleast 2 classes")

        train_cls = [i for i in range(ll.c - 1) if i not in hidden_classes]

        def mask(i, c): i.y.items == c

        # Copy to get transforms
        tll = deepcopy(ll)
        # Generate Items and Labels
        tll.x.items = np.concatenate(
            [ll.x.items[mask(ll, c)] for c in train_cls])
        tll.y.items = np.concatenate(
            [ll.y.items[mask(ll, c)] for c in train_cls])

        vll = deepcopy(ll)
        vll.x.items = np.concatenate(
            [ll.x.items[mask(ll, c)] for c in hidden_classes])
        vll.y.items = np.concatenate(
            [ll.y.items[mask(ll, c)] for c in hidden_classes])

        tll = cls._from_label_list(tll, train_num, train_num//2, use_all)
        vll = cls._from_label_list(vll, valid_num, valid_num//2, use_all)
        return LabelLists(ll.x.path, tll, vll)

    @classmethod
    def _from_label_list(cls, ll: LabelList, num_same=20, num_diff=30, use_all=False):
        x = ll.x
        y = ll.y
        # Seperate into lists of individual classes
        # Some of these may be empty because of the spit
        seperated = [x.items[y.items == c]
                     for c in range(ll.c) if len(x.items[y.items == c]) > 0]

        # Create sets of same pairs
        # TODO Don't create the same pairs
        # if use_all:

        same_pairs = np.empty((0, 2))
        for cis in seperated:
            r = np.array([np.random.choice(cis, num_same),
                          np.random.choice(cis, num_same)]).T
            same_pairs = np.concatenate([same_pairs, r])

        # Create pairs of different items
        diff_pairs = np.empty((0, 2))

        for i, cis in enumerate(seperated):
            other = [k for k in range(len(seperated)) if k != i]
            for i in other:
                ocis = seperated[i]
                dps = np.array([np.random.choice(cis, num_diff),
                                np.random.choice(ocis, num_diff)]).T
                diff_pairs = np.concatenate([diff_pairs, dps])

        # Combine together
        al = np.concatenate([same_pairs, diff_pairs])
        # Generate labels
        labels = np.concatenate([np.ones(len(same_pairs), dtype=np.int8), np.zeros(
            len(diff_pairs), dtype=np.int8)])

        inst = cls(SiameseList(al, open_fn=ll.x.open),
                   CategoryList(labels, ['different', 'similar']))
        return inst
