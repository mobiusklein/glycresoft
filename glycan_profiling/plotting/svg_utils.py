# try:   # pragma: no cover
#     from cStringIO import StringIO
# except:  # pragma: no cover
#     try:
#         from StringIO import StringIO
#     except:
#         from io import StringIO
# from io import StringIO
from io import BytesIO
try:  # pragma: no cover
    from lxml import etree as ET
except ImportError:  # pragma: no cover
    try:
        from xml.etree import cElementTree as ET
    except ImportError:
        from xml.etree import ElementTree as ET


class IDMapper(dict):
    '''
    A dictionary-like container which uses a format-string
    key pattern to generate unique identifiers for each entry

    Key Pattern: '<type-name>-%d'

    Associates each generated id with a dictionary of metadata and
    sets the `gid` of the passed `matplotlib.Artist` to the generated
    id. Only the id and metadata are stored.

    Used to preserve a mapping of metadata to artists for later SVG
    serialization.
    '''

    def __init__(self):
        dict.__init__(self)
        self.counter = 0

    def add(self, key, value, meta):
        label = key % self.counter
        value.set_gid(label)
        self[label] = meta
        self.counter += 1
        return label
