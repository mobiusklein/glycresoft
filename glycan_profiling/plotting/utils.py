from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def figax(*args, **kwargs):
    fig = Figure(*args, **kwargs)
    canvas = FigureCanvasAgg(fig)
    return fig.add_subplot(1, 1, 1)


def figure(*args, **kwargs):
    fig = Figure(*args, **kwargs)
    canvas = FigureCanvasAgg(fig)
    return fig
