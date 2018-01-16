from glycan_profiling.structure import KeyTransformingDecoratorDict


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
