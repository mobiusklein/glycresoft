class SpectrumReference(object):
    def __init__(self, scan_id, precursor_information=None):
        self.scan_id = scan_id
        self.precursor_information = precursor_information

    @property
    def id(self):
        return self.scan_id

    def __repr__(self):
        return "SpectrumReference({self.id})".format(self=self)


class TargetReference(object):
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return "TargetReference({self.id})".format(self=self)
