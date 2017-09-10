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


def key_transform(name):
    return str(name).lower().replace(" ", '-')


hypothesis_register = KeyTransformingDecoratorDict(key_transform)


class BuildBase(object):
    def get_hypothesis_metadata(self):
        raise NotImplementedError()

    @property
    def hypothesis_metadata(self):
        return self.get_hypothesis_metadata()

    def build(self, database_connection, **kwargs):
        raise NotImplementedError()

    def __call__(self, database_connection, **kwargs):
        return self.build(database_connection, **kwargs)
