ms1_model_features = dict()


def register_feature(name, feature):
    ms1_model_features[name] = feature


def available_features():
    return list(ms1_model_features)


def get_feature(name):
    return ms1_model_features[name]
