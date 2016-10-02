import re
import logging

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


class Parser(MzIdentML):

    def _retrieve_refs(self, info, **kwargs):
        """Retrieves and embeds the data for each attribute in `info` that
        ends in _ref. Removes the id attribute from `info`"""
        multi = None
        for k, v in dict(info).items():
            if k.endswith('_ref'):
                try:
                    info.update(self.get_by_id(v, retrieve_refs=True))
                    del info[k]
                    info.pop('id', None)
                except MultipleProteinMatchesException, e:
                    multi = e
                except:
                    is_multi_db_sequence = peptide_evidence_ref.match(info[k])
                    if is_multi_db_sequence:
                        groups = is_multi_db_sequence.groupdict()
                        evidence_id = groups['evidence_id']
                        db_sequences = groups['parent_accession'].split(':')
                        if len(db_sequences) > 1:
                            multi = MultipleProteinMatchesException(
                                "", evidence_id, db_sequences, k)
                            continue
                    # Fall through
                    logger.debug("%s not found", v)
                    info['skip'] = True
                    info[k] = v
        if multi is not None:
            raise multi

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

    def _get_info(self, element, **kwargs):
        """Extract info from element's attributes, possibly recursive.
        <cvParam> and <userParam> elements are treated in a special way."""
        name = _local_name(element)
        schema_info = self.schema_info
        if name in {'cvParam', 'userParam'}:
            return self._handle_param(element)

        info = dict(element.attrib)
        # process subelements
        if kwargs.get('recursive'):
            for child in element.iterchildren():
                cname = _local_name(child)
                if cname in {'cvParam', 'userParam'}:
                    self._insert_param(info, child, **kwargs)
                else:
                    if cname not in schema_info['lists']:
                        info[cname] = self._get_info_smart(child, **kwargs)
                    else:
                        info.setdefault(cname, []).append(
                            self._get_info_smart(child, **kwargs))
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

        # convert types
        converters = self._converters
        for k, v in info.items():
            for t, a in converters.items():
                if (_local_name(element), k) in schema_info[t]:
                    info[k] = a(v)
        infos = [info]
        try:
            # resolve refs
            if kwargs.get('retrieve_refs'):
                self._retrieve_refs(info, **kwargs)
        except MultipleProteinMatchesException, e:
            evidence_id = e.evidence_id
            db_sequences = e.db_sequences
            key = e.key
            infos = []
            for name in db_sequences:
                dup = info.copy()
                dup[key] = evidence_id + name
                self._retrieve_refs(dup, **kwargs)
                infos.append(dup)

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
