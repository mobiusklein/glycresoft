import codecs
import pkg_resources


def make_model_loader(model_type):
    def load_model(model_name):
        stream = codecs.getreader("utf-8")(
            pkg_resources.resource_stream(
                __name__, "data/%s.json" % model_name))
        return model_type.load(stream)
    return load_model
