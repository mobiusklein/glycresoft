from glycopeptidepy.utils.collectiontools import decoratordict


class KeyTransformingDecoratorDict(decoratordict):
    def __init__(self, transform, *args, **kwargs):
        self.transform = transform
        super(KeyTransformingDecoratorDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        return super(KeyTransformingDecoratorDict, self).__getitem__(self.transform(key))

    def __setitem__(self, key, value):
        key = self.transform(key)
        super(KeyTransformingDecoratorDict, self).__setitem__(key, value)

    def __contains__(self, key):
        key = self.transform(key)
        return dict.__contains__(self, key)

    def keys(self, transform=None):
        if transform is None:
            transform = self.transform
        return TransformingView(transform, super(KeyTransformingDecoratorDict, self).keys())


class TransformingView(object):
    def __init__(self, transform, values):
        self.transform = transform
        self.values = tuple(map(self.transform, values))

    def __contains__(self, key):
        key = self.transform(key)
        return key in self.values

    def __iter__(self):
        return iter(self.values)

    def __repr__(self):
        return "TransformingView%r" % (self.values,)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]
