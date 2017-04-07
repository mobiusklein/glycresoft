from .utils import figax


class ArtistBase(object):

    def __init__(self, ax=None):
        self.ax = ax

    def create_axes(self):
        self.ax = figax()

    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)

    def _repr_html_(self):
        if self.ax is None:
            return repr(self)
        fig = (self.ax.get_figure())
        return fig._repr_html_()
