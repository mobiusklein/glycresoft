import urllib
import logging

from io import BytesIO

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib import rcParams as mpl_params

from lxml import etree

import jinja2
from jinja2 import escape

from glypy.structure.glycan_composition import GlycanComposition
from glycopeptidepy import PeptideSequence

from glycan_profiling.task import TaskBase
from glycan_profiling.serialize import DatabaseBoundOperation
from glycan_profiling.symbolic_expression import GlycanSymbolContext
from glycan_profiling.plotting import colors


mpl_params.update({
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'font.size': 10,
    'savefig.dpi': 72,
    'figure.subplot.bottom': .125})


status_logger = logging.getLogger("glycresoft.status")


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


def xmlattrs(**kwargs):
    return ' '.join("%s=\"%s\"" % kv for kv in kwargs.items())


def png_plot(figure, img_width=None, img_height=None, xattrs=None, **kwargs):
    if xattrs is None:
        xattrs = dict()
    xml_attributes = dict(xattrs)
    if img_width is not None:
        xml_attributes['width'] = img_width
    if img_height is not None:
        xml_attributes['height'] = img_height
    data_buffer = render_plot(figure, format='png', **kwargs)
    return "<img %s src='data:image/png;base64,%s'>" % (
        xmlattrs(**xml_attributes),
        urllib.quote(data_buffer.getvalue().encode("base64")))


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


def rgbpack(color):
    return "rgba(%d,%d,%d,0.5)" % tuple(i * 255 for i in color)


def formula(composition):
    return ''.join("<b>%s</b><sub>%d</sub>" % (k, v) for k, v in sorted(composition.items()))


def glycan_composition_string(composition):
    try:
        composition = GlycanComposition.parse(
            GlycanSymbolContext(
                GlycanComposition.parse(
                    composition)).serialize())
    except ValueError:
        return "<code>%s</code>" % composition

    parts = []
    template = ("<span class='monosaccharide-composition-name'"
                "style='background-color:%s'>"
                "%s&nbsp;%d</span>")
    for k, v in sorted(composition.items(), key=lambda x: x[0].mass()):
        name = str(k)
        color = colors.get_color(str(name))
        parts.append(template % (rgbpack(color), name, v))
    reduced = composition.reducing_end
    if reduced:
        reducing_end_template = (
            "<span class='monosaccharide-composition-name'"
            "style='background-color:%s;padding:2px;border-radius:2px;'>"
            "%s</span>")
        name = formula(reduced.composition)
        color = colors.get_color(str(name))
        parts.append(reducing_end_template % (rgbpack(color), name))

    return ' '.join(parts)


def glycopeptide_string(sequence, long=False, include_glycan=True):
    sequence = PeptideSequence(str(sequence))
    parts = []
    template = "(<span class='modification-chip'"\
        " style='background-color:%s;padding-left:1px;padding-right:2px;border-radius:2px;'"\
        " title='%s' data-modification='%s'>%s</span>)"

    n_term_template = template.replace("(", "").replace(")", "") + '-'
    c_term_template = "-" + (template.replace("(", "").replace(")", ""))

    def render(mod, template=template):
        color = colors.get_color(str(mod))
        letter = escape(mod.name if long else mod.name[0])
        name = escape(mod.name)
        parts.append(template % (rgbpack(color), name, name, letter))

    if sequence.n_term != "H":
        render(sequence.n_term, n_term_template)
    for res, mods in sequence:
        parts.append(res.symbol)
        for mod in mods:
            render(mod)
    if sequence.c_term != "OH":
        render(sequence.c_term, c_term_template)
    parts.append((
        ' ' + glycan_composition_string(str(sequence.glycan)) if sequence.glycan is not None else "")
        if include_glycan else "")
    return ''.join(parts)


def highlight_sequence_site(amino_acid_sequence, site_list, site_type_list):
    if isinstance(site_type_list, basestring):
        site_type_list = [site_type_list for i in site_list]
    sequence = list(amino_acid_sequence)
    for site, site_type in zip(site_list, site_type_list):
        sequence[site] = "<span class='{}'>{}</span>".format(site_type, sequence[site])
    return sequence


def n_per_row(sequence, n=60):
    row_buffer = []
    i = 0
    while i < len(sequence):
        row_buffer.append(
            ''.join(sequence[i:(i + n)])
        )
        i += n
    return '<br>'.join(row_buffer)


class ReportCreatorBase(TaskBase):
    def __init__(self, database_connection, analysis_id, stream=None):
        self.database_connection = DatabaseBoundOperation(database_connection)
        self.analysis_id = analysis_id
        self.stream = stream
        self.env = jinja2.Environment()

    @property
    def session(self):
        return self.database_connection.session

    def status_update(self, message):
        status_logger.info(message)

    def prepare_environment(self):
        self.env.filters['svguri_plot'] = svguri_plot
        self.env.filters['png_plot'] = png_plot
        self.env.filters['svg_plot'] = svg_plot
        self.env.filters["n_per_row"] = n_per_row
        self.env.filters['highlight_sequence_site'] = highlight_sequence_site
        self.env.filters['svg_plot'] = svg_plot
        self.env.filters['glycopeptide_string'] = glycopeptide_string
        self.env.filters['glycan_composition_string'] = glycan_composition_string
        self.env.filters["formula"] = formula

    def set_template_loader(self, path):
        self.env.loader = jinja2.FileSystemLoader(path)

    def make_template_stream(self):
        raise NotImplementedError()

    def run(self):
        self.prepare_environment()
        template_stream = self.make_template_stream()
        template_stream.dump(self.stream, encoding='utf-8')
