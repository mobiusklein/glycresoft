from itertools import cycle
from matplotlib.colors import cnames, hex2color, rgb_to_hsv, hsv_to_rgb
from matplotlib import patches as mpatches


def lighten(rgb, factor=0.25):
    '''Given a triplet of rgb values, lighten the color by `factor`%'''
    factor += 1
    return [min(c * factor, 1) for c in rgb]


def darken(rgb, factor=0.25):
    '''Given a triplet of rgb values, darken the color by `factor`%'''
    factor = 1 - factor
    return [(c * factor) for c in rgb]


def deepen(rgb, factor=0.25):
    hsv = rgb_to_hsv(rgb)
    hsv[1] = min(hsv[1] * (1 + factor), 1)
    return hsv_to_rgb(hsv)


colors = cycle([hex2color(cnames[name]) for name in (
    "red", "blue", "yellow", "purple", "navy", "grey", "coral", "forestgreen", "limegreen", "maroon", "aqua",
    "lavender", "lightcoral", "mediumorchid")])


class ColorMapper(object):
    colors = [hex2color(cnames[name]) for name in (
              "red", "blue", "yellow", "purple", "navy", "grey", "coral", "forestgreen", "limegreen", "maroon", "aqua",
              "lavender", "lightcoral", "mediumorchid")]

    def __init__(self):
        self.color_name_map = {
            "HexNAc": hex2color(cnames["mediumseagreen"]),
            "N-Glycosylation": hex2color(cnames["mediumseagreen"]),
            "O-Glycosylation": hex2color(cnames["cadetblue"]),
            "GAG-Linker": hex2color(cnames['burlywood']),
            "Xyl": hex2color(cnames['darkorange']),
            "a,enHex": hex2color(cnames['lightskyblue']),
            "aHex": hex2color(cnames['plum']),
            "Hex": hex2color(cnames['steelblue']),
            "Neu5Ac": hex2color(cnames['blueviolet']),
            "Neu5Gc": hex2color(cnames['lightsteelblue']),
            "Fuc": hex2color(cnames['crimson'])
        }
        self.color_generator = cycle(self.colors)

    def get_color(self, name):
        """Given a name, find the color mapped to that name, or
        select the next color from the `colors` generator and assign
        it to the name and return the new color.

        Parameters
        ----------
        name : object
            Any hashable object, usually a string

        Returns
        -------
        tuple: RGB triplet
        """
        try:
            return self.color_name_map[name]
        except KeyError:
            o = self.color_name_map[name] = next(self.color_generator)
            return o

    __getitem__ = get_color

    def __setitem__(self, name, color):
        self.color_name_map[name] = color

    def __repr__(self):
        return repr(self.color_name_map)

    darken = staticmethod(darken)
    lighten = staticmethod(lighten)

    def keys(self):
        return self.color_name_map.keys()

    def items(self):
        return self.color_name_map.items()

    def proxy_artists(self, subset=None):
        proxy_artists = []
        if subset is not None:
            for name, color in self.items():
                if name in subset:
                    artist = mpatches.Rectangle((0, 0), 1, 1, fc=color, label=name)
                    proxy_artists.append((name, artist))
        else:
            for name, color in self.items():
                artist = mpatches.Rectangle((0, 0), 1, 1, fc=color, label=name)
                proxy_artists.append((name, artist))
        return proxy_artists


_color_mapper = ColorMapper()

color_name_map = _color_mapper.color_name_map
get_color = _color_mapper.get_color
proxy_artists = _color_mapper.proxy_artists


def color_dict():
    return {str(k): v for k, v in _color_mapper.items()}


material_palette = [(0.9568627450980393, 0.2627450980392157, 0.21176470588235294),
                    (0.24705882352941178, 0.3176470588235294, 0.7098039215686275),
                    (0.2980392156862745, 0.6862745098039216, 0.3137254901960784),
                    (1.0, 0.596078431372549, 0.0),
                    (0.9137254901960784, 0.11764705882352941, 0.38823529411764707),
                    (0.12941176470588237, 0.5882352941176471, 0.9529411764705882),
                    (0.5450980392156862, 0.7647058823529411, 0.2901960784313726),
                    (1.0, 0.3411764705882353, 0.13333333333333333),
                    (0.011764705882352941, 0.6627450980392157, 0.9568627450980393),
                    (0.803921568627451, 0.8627450980392157, 0.2235294117647059),
                    (0.4745098039215686, 0.3333333333333333, 0.2823529411764706),
                    (0.611764705882353, 0.15294117647058825, 0.6901960784313725),
                    (0.0, 0.7372549019607844, 0.8313725490196079),
                    (1.0, 0.9215686274509803, 0.23137254901960785),
                    (0.6196078431372549, 0.6196078431372549, 0.6196078431372549),
                    (0.403921568627451, 0.22745098039215686, 0.7176470588235294),
                    (0.0, 0.5882352941176471, 0.5333333333333333),
                    (1.0, 0.7568627450980392, 0.027450980392156862),
                    (0.3764705882352941, 0.49019607843137253, 0.5450980392156862)]
