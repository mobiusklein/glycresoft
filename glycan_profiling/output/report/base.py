import urllib
from io import BytesIO

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib import rcParams as mpl_params

from lxml import etree

import jinja2

from glycan_profiling.task import TaskBase
from glycan_profiling.serialize import DatabaseBoundOperation


mpl_params.update({
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'font.size': 10,
    'savefig.dpi': 72,
    'figure.subplot.bottom': .125})


def render_plot(figure, **kwargs):
    if isinstance(figure, Axes):
        figure = figure.get_figure()
    if "height" in kwargs:
        figure.set_figheight(kwargs["height"])
    if "width" in kwargs:
        figure.set_figwidth(kwargs['width'])
    if kwargs.get("bbox_inches") != 'tight' or kwargs.get("patchless"):
        figure.patch.set_alpha(0)
        figure.axes[0].patch.set_alpha(0)
    data_buffer = BytesIO()
    figure.savefig(data_buffer, **kwargs)
    plt.close(figure)
    return data_buffer


def png_plot(figure, **kwargs):
    data_buffer = render_plot(figure, format='png', **kwargs)
    return "<img src='data:image/png;base64,%s'>" % urllib.quote(data_buffer.getvalue().encode("base64"))


def svguri_plot(figure, **kwargs):
    svg_string = svg_plot(figure, **kwargs)
    return "<img src='data:image/svg+xml;utf-8,%s'>" % urllib.quote(svg_string)


def _strip_style(root):
    style_node = root.find(".//{http://www.w3.org/2000/svg}style")
    style_node.text = ""
    return root


def svg_plot(figure, svg_width=None, xml_transform=None, **kwargs):
    data_buffer = render_plot(figure, format='svg', **kwargs)
    root = etree.fromstring(data_buffer.getvalue())
    root = _strip_style(root)
    if svg_width is not None:
        root.attrib["width"] = svg_width
    if xml_transform is not None:
        root = xml_transform(root)
    return etree.tostring(root)


class ReportCreatorBase(TaskBase):
    def __init__(self, database_connection, analysis_id, stream=None):
        self.database_connection = DatabaseBoundOperation(database_connection)
        self.analysis_id = analysis_id
        self.stream = stream
        self.env = jinja2.Environment()

    def prepare_environment(self):
        self.env.filters['svguri_plot'] = svguri_plot
        self.env.filters['png_plot'] = png_plot

    def set_template_loader(self, path):
        self.env.loader = jinja2.FileSystemLoader(path)

    def make_template_stream(self):
        raise NotImplementedError()

    def run(self):
        self.prepare_environment()
        template_stream = self.make_template_stream()
        template_stream.dump(self.stream)
