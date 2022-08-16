from glycan_profiling.structure import KeyTransformingDecoratorDict


def transform_key(key):
    return key.replace("-", "_")


ms1_model_features = KeyTransformingDecoratorDict(transform_key)


def register_feature(name, feature):
    ms1_model_features[name] = feature


def available_features():
    return list(ms1_model_features)


def get_feature(name):
    return ms1_model_features[name]
