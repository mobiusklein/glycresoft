try:   # pragma: no cover
    from cStringIO import StringIO
except:  # pragma: no cover
    try:
        from StringIO import StringIO
    except:
        from io import StringIO
try:  # pragma: no cover
    from lxml import etree as ET
except ImportError:  # pragma: no cover
    try:
        from xml.etree import cElementTree as ET
    except:
        from xml.etree import ElementTree as ET
