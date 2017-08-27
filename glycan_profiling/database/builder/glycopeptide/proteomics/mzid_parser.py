import re
import logging
from lxml.etree import LxmlError
from pyteomics import mzid

logger = logging.getLogger("mzid")

MzIdentML = mzid.MzIdentML
_local_name = mzid.xml._local_name
peptide_evidence_ref = re.compile(r"(?P<evidence_id>PEPTIDEEVIDENCE_PEPTIDE_\d+_DBSEQUENCE_)(?P<parent_accession>.+)")


class MultipleProteinMatchesException(Exception):

    def __init__(self, message, evidence_id, db_sequences, key):
        Exception.__init__(self, message)
        self.evidence_id = evidence_id
        self.db_sequences = db_sequences
        self.key = key


class MultipleProteinInfoDict(dict):
    multi = None


class MissingProteinException(Exception):
    pass


class MissingPeptideEvidenceHandler(object):
    parser = re.compile(
        r"(?P<evidence_id>PEPTIDEEVIDENCE_(?P<peptide_id>PEPTIDE_\d+)_(?P<parent_accession>DBSEQUENCE_.+))")

    def __init__(self, full_id):
        self.full_id = full_id

    def __repr__(self):
        return "MissingPeptideEvidenceHandler(%r)" % (self.full_id,)

    def reconstruct_evidence(self):
        match = self.parser.search(self.full_id)
        if match is None:
            raise ValueError("Cannot interpret %r" % (self.full_id,))
        data = match.groupdict()
        result = {
            "isDecoy": False,
            "end": None,
            "start": None,
            "peptide_ref": data["peptide_id"],
            "dBSequence_ref": data['parent_accession'],
            'pre': None,
            "post": None,
            "id": self.full_id
        }
        return result

    @classmethod
    def recover(cls, full_id):
        inst = cls(full_id)
        return inst.reconstruct_evidence()


class Parser(MzIdentML):
    def _handle_ref(self, info, key, value):
        try:
            referenced = self.get_by_id(value, retrieve_refs=True)
        except AttributeError:
            if key == "peptideEvidence_ref":
                referenced = MissingPeptideEvidenceHandler.recover(value)
            else:
                raise AttributeError(key)
        info.update(referenced)
        del info[key]
        info.pop('id', None)

    def _retrieve_refs(self, info, **kwargs):
        multi = None
        for k, v in dict(info).items():
            if k.endswith('_ref'):
                is_multi_db_sequence = peptide_evidence_ref.match(info[k])
                if is_multi_db_sequence and ':' in is_multi_db_sequence.groupdict()['parent_accession']:
                    groups = is_multi_db_sequence.groupdict()
                    evidence_id = groups['evidence_id']
                    db_sequences = groups['parent_accession'].split(':')
                    if len(db_sequences) > 1:
                        multi = MultipleProteinMatchesException(
                            "", evidence_id, db_sequences, k)
                else:
                    try:
                        self._handle_ref(info, k, v)
                    except (KeyError, LxmlError):
                        info['skip'] = True
        info = MultipleProteinInfoDict(info)
        info.multi = multi
        return info, multi

    def _find_by_id_reset(self, *a, **kw):
        return MzIdentML._find_by_id_reset(self, *a, **kw)

    def _insert_param(self, info, param, **kwargs):
        newinfo = self._handle_param(param, **kwargs)
        if not ('name' in info and 'name' in newinfo):
            info.update(newinfo)
        else:
            if not isinstance(info['name'], list):
                info['name'] = [info['name']]
            info['name'].append(newinfo.pop('name'))

    def _find_immediate_params(self, element, **kwargs):
        return element.xpath('./*[local-name()="{}" or local-name()="{}"]'.format("cvParam", "userParam"))

    def _recursive_populate(self, element, info, kwargs):
        for child in element.iterchildren():
            cname = _local_name(child)
            if cname in ('cvParam', 'userParam'):
                self._insert_param(info, child, **kwargs)
            else:
                if cname not in self.schema_info['lists']:
                    info[cname] = self._get_info_smart(child, **kwargs)
                else:
                    info.setdefault(cname, []).append(
                        self._get_info_smart(child, **kwargs))

    def _convert_values(self, element, info, kwargs):
        converters = self._converters
        for k, v in info.items():
            try:
                for t, a in converters.items():
                    if (_local_name(element), k) in self.schema_info[t]:
                        info[k] = a(v)
            except KeyError:
                continue

    def _populate_references(self, element, info, kwargs):
        info = MultipleProteinInfoDict(info)
        infos = [info]
        # resolve refs
        if kwargs.get('retrieve_refs'):
            try:
                _, multi = self._retrieve_refs(info, **kwargs)
            except ValueError as e:
                print(info, e, kwargs)
                raise e
            if multi is not None:
                info.multi = multi
            if info.multi:
                e = info.multi
                if e is not None:
                    evidence_id = e.evidence_id
                    db_sequences = e.db_sequences
                    key = e.key
                    infos = []
                    for name in db_sequences:
                        dup = info.copy()
                        dup[key] = evidence_id + name
                        self._retrieve_refs(dup, **kwargs)
                        infos.append(dup)
        return infos

    def _get_info(self, element, **kwargs):
        """Extract info from element's attributes, possibly recursive.
        <cvParam> and <userParam> elements are treated in a special way."""
        name = _local_name(element)
        if name in {'cvParam', 'userParam'}:
            return self._handle_param(element)

        info = dict(element.attrib)
        # process subelements
        if kwargs.get('recursive'):
            self._recursive_populate(element, info, kwargs)
        else:
            for param in self._find_immediate_params(element):
                self._insert_param(info, param, **kwargs)

        # process element text
        if element.text and element.text.strip():
            stext = element.text.strip()
            if stext:
                if info:
                    info[name] = stext
                else:
                    return stext

        self._convert_values(element, info, kwargs)

        infos = self._populate_references(element, info, kwargs)

        # flatten the excessive nesting
        for info in infos:
            for k, v in dict(info).items():
                if k in self._structures_to_flatten:
                    info.update(v)
                    del info[k]

            # another simplification
            for k, v in dict(info).items():
                if isinstance(v, dict) and 'name' in v and len(v) == 1:
                    info[k] = v['name']
        out = []
        for info in infos:
            if len(info) == 2 and 'name' in info and (
                    'value' in info or 'values' in info):
                name = info.pop('name')
                info = {name: info.popitem()[1]}
            out.append(info)
        if len(out) == 1:
            out = out[0]
        return out
