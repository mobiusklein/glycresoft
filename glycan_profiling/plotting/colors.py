from itertools import cycle
from matplotlib.colors import cnames, hex2color
from matplotlib import patches as mpatches


def lighten(rgb, factor=0.25):
    '''Given a triplet of rgb values, lighten the color by `factor`%'''
    factor += 1
    return [min(c * factor, 1) for c in rgb]


def darken(rgb, factor=0.25):
    '''Given a triplet of rgb values, darken the color by `factor`%'''
    factor = 1 - factor
    return [(c * factor) for c in rgb]


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
            o = self.color_name_map[name] = self.color_generator.next()
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
