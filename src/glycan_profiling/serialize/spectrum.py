from typing import Dict, Optional
from sqlalchemy.orm.session import object_session

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, select)
from sqlalchemy.orm import relationship, backref, deferred
from sqlalchemy.ext.mutable import MutableDict, MutableList

import numpy as np

from ms_peak_picker import FittedPeak as MemoryFittedPeak, PeakIndex, PeakSet
from ms_deisotope import DeconvolutedPeak as MemoryDeconvolutedPeak, DeconvolutedPeakSet
from ms_deisotope.peak_set import Envelope
from ms_deisotope.averagine import (
    mass_charge_ratio)

from ms_deisotope.data_source.common import (
    ProcessedScan, PrecursorInformation as MemoryPrecursorInformation, ChargeNotProvided)


from .base import (Base, Mass, HasUniqueName)
from .utils import chunkiter


class SampleRun(Base, HasUniqueName):
    __tablename__ = "SampleRun"

    id = Column(Integer, primary_key=True, autoincrement=True)

    ms_scans = relationship(
        "MSScan", backref=backref("sample_run"), lazy='dynamic')
    sample_type = Column(String(128))

    completed = Column(Boolean(), default=False, nullable=False)

    def __repr__(self):
        return "SampleRun(id=%d, name=%s)" % (self.id, self.name)


class MSScan(Base):
    __tablename__ = "MSScan"

    id = Column(Integer, primary_key=True, autoincrement=True)
    index = Column(Integer, index=True)
    ms_level = Column(Integer)
    scan_time = Column(Numeric(10, 5, asdecimal=False), index=True)
    title = Column(String(512))
    scan_id = Column(String(512), index=True)
    sample_run_id = Column(Integer, ForeignKey(
        SampleRun.id, ondelete='CASCADE'), index=True)

    peak_set = relationship("FittedPeak", backref="scan", lazy="dynamic")
    deconvoluted_peak_set = relationship(
        "DeconvolutedPeak", backref='scan', lazy='dynamic')

    info = deferred(Column(MutableDict.as_mutable(PickleType)))

    def __repr__(self):
        f = "{}({}, {}, {}, {}".format(
            self.__class__.__name__, self.scan_id, self.ms_level, self.scan_time,
            self.deconvoluted_peak_set.count())
        if self.ms_level > 1:
            f = "%s %s" % (f, self.precursor_information)
        f += ")"
        return f

    def convert(self, fitted=False, deconvoluted=True):
        precursor_information = self.precursor_information
        # if precursor_information is not None:
        #     precursor_information = precursor_information.convert()


        session = object_session(self)
        conn = session.connection()

        if fitted:
            q = conn.execute(select([FittedPeak.__table__]).where(
                FittedPeak.__table__.c.scan_id == self.id)).fetchall()

            peak_set_items = list(
                map(make_memory_fitted_peak, q))

            peak_set = PeakSet(peak_set_items)
            peak_set._index()
            peak_index = PeakIndex(np.array([], dtype=np.float64), np.array(
                [], dtype=np.float64), peak_set)
        else:
            peak_index = PeakIndex(np.array([], dtype=np.float64), np.array(
                [], dtype=np.float64), PeakSet([]))

        if deconvoluted:
            q = conn.execute(select([DeconvolutedPeak.__table__]).where(
                DeconvolutedPeak.__table__.c.scan_id == self.id)).fetchall()

            deconvoluted_peak_set_items = list(
                map(make_memory_deconvoluted_peak, q))

            deconvoluted_peak_set = DeconvolutedPeakSet(
                deconvoluted_peak_set_items)
            deconvoluted_peak_set.reindex()
        else:
            deconvoluted_peak_set = DeconvolutedPeakSet([])

        info = self.info or {}

        (scan_id, scan_title, scan_ms_level, scan_time, scan_index) = session.query(
            MSScan.scan_id, MSScan.title, MSScan.ms_level, MSScan.scan_time, MSScan.index).filter(
                MSScan.id == self.id).first()

        scan = ProcessedScan(
            scan_id, scan_title, precursor_information, int(scan_ms_level),
            float(scan_time), scan_index, peak_index, deconvoluted_peak_set,
            activation=info.get('activation'))
        return scan

    @classmethod
    def bulk_load(cls, session, ids, chunk_size: int=512, scan_cache: Optional[Dict]=None):
        if scan_cache is None:
            scan_cache = {}
        out = scan_cache

        for chunk in chunkiter(ids, chunk_size):
            id_slice = list(filter(lambda x: x not in scan_cache,
                                   set(chunk)))
            res = session.execute(cls.__table__.select().where(
                cls.id.in_(id_slice)))
            pinfos = PrecursorInformation.bulk_load_from_product_ids(
                session, id_slice, chunk_size=chunk_size)
            for self in res.fetchall():
                peak_index = None
                deconvoluted_peak_set = DeconvolutedPeakSet([])
                scan = ProcessedScan(
                    self.scan_id, self.scan_id, pinfos.get(self.id), int(self.ms_level),
                    float(self.scan_time), self.index, peak_index, deconvoluted_peak_set,
                    activation=self.info.get('activation'))
                out[self.id] = scan
        return [out[i] for i in ids]

    @classmethod
    def _serialize_scan(cls, scan, sample_run_id=None):
        db_scan = cls(
            index=scan.index, ms_level=scan.ms_level,
            scan_time=float(scan.scan_time), title=scan.title,
            scan_id=scan.id, sample_run_id=sample_run_id,
            info={'activation': scan.activation})
        return db_scan

    @classmethod
    def serialize(cls, scan, sample_run_id=None):
        db_scan = cls._serialize_scan(scan, sample_run_id)
        db_scan.peak_set = map(FittedPeak.serialize, scan.peak_set)
        db_scan.deconvoluted_peak_set = map(
            DeconvolutedPeak.serialize, scan.deconvoluted_peak_set)
        return db_scan

    @classmethod
    def serialize_bulk(cls, scan, sample_run_id, session, fitted=True, deconvoluted=True):
        db_scan = cls._serialize_scan(scan, sample_run_id)

        session.add(db_scan)
        session.flush()

        if fitted:
            FittedPeak._serialize_bulk_list(scan.peak_set, db_scan.id, session)
        if deconvoluted:
            DeconvolutedPeak._serialize_bulk_list(
                scan.deconvoluted_peak_set, db_scan.id, session)
        return db_scan


class PrecursorInformation(Base):
    __tablename__ = "PrecursorInformation"

    id = Column(Integer, primary_key=True)
    sample_run_id = Column(Integer, ForeignKey(
        SampleRun.id, ondelete='CASCADE'), index=True)

    precursor_id = Column(Integer, ForeignKey(
        MSScan.id, ondelete='CASCADE'), index=True)
    precursor = relationship(MSScan, backref=backref("product_information"),
                             primaryjoin="PrecursorInformation.precursor_id == MSScan.id",
                             uselist=False)
    product_id = Column(Integer, ForeignKey(
        MSScan.id, ondelete='CASCADE'), index=True)
    product = relationship(MSScan, backref=backref("precursor_information", uselist=False),
                           primaryjoin="PrecursorInformation.product_id == MSScan.id",
                           uselist=False)
    neutral_mass = Mass()
    charge = Column(Integer)
    intensity = Column(Numeric(16, 4, asdecimal=False))
    defaulted = Column(Boolean)
    orphan = Column(Boolean)

    @property
    def extracted_neutral_mass(self):
        return self.neutral_mass

    @property
    def extracted_charge(self):
        return self.charge

    @property
    def extracted_intensity(self):
        return self.intensity

    def __repr__(self):
        return "DBPrecursorInformation({}, {}, {})".format(
            self.precursor.scan_id, self.neutral_mass, self.charge)

    def convert(self, data_source=None):
        precursor_id = None
        if self.precursor is not None:
            precursor_id = self.precursor.scan_id
        product_id = None
        if self.product is not None:
            product_id = self.product.scan_id
        charge = self.charge
        if charge == 0:
            charge = ChargeNotProvided
        conv = MemoryPrecursorInformation(
            mass_charge_ratio(self.neutral_mass, charge), self.intensity, charge,
            precursor_id, data_source, self.neutral_mass, charge,
            self.intensity, self.defaulted, self.orphan, product_scan_id=product_id)
        return conv

    @classmethod
    def bulk_load_from_product_ids(cls, session, ids, chunk_size: int=512, data_source=None) -> Dict[int, MemoryPrecursorInformation]:
        out = {}

        for chunk in chunkiter(ids, chunk_size):
            res = session.execute(cls.__table__.select().where(cls.product_id.in_(chunk)))
            for self in res.fetchall():
                precursor_id = None
                if self.precursor_id is not None:
                    precursor_id = session.execute(select([MSScan.scan_id]).where(
                        MSScan.id == self.precursor_id)).scalar()

                product_id = None
                if self.product_id is not None:
                    product_id = session.execute(select([MSScan.scan_id]).where(
                        MSScan.id == self.product_id)).scalar()

                charge = self.charge
                if charge == 0:
                    charge = ChargeNotProvided

                conv = MemoryPrecursorInformation(
                    mass_charge_ratio(self.neutral_mass, charge), self.intensity, charge,
                    precursor_id, data_source, self.neutral_mass, charge,
                    self.intensity, self.defaulted, self.orphan, product_scan_id=product_id)
                out[self.product_id] = conv
        return out

    @classmethod
    def serialize(cls, inst, precursor, product, sample_run_id):
        charge = inst.extracted_charge
        if charge == ChargeNotProvided:
            charge = 0
        db_pi = PrecursorInformation(
            precursor_id=precursor.id, product_id=product.id,
            charge=charge, intensity=inst.extracted_intensity,
            neutral_mass=inst.extracted_neutral_mass, sample_run_id=sample_run_id,
            defaulted=inst.defaulted, orphan=inst.orphan)
        return db_pi


class PeakMixin(object):
    mz = Mass(False)
    intensity = Column(Numeric(16, 4, asdecimal=False))
    full_width_at_half_max = Column(Numeric(7, 6, asdecimal=False))
    signal_to_noise = Column(Numeric(10, 3, asdecimal=False))
    area = Column(Numeric(12, 4, asdecimal=False))

    @classmethod
    def _serialize_bulk_list(cls, peaks, scan_id, session):
        out = cls._prepare_serialize_list(peaks, scan_id)
        session.bulk_save_objects(out)

    @classmethod
    def _prepare_serialize_list(cls, peaks, scan_id):
        out = []
        for peak in peaks:
            db_peak = cls.serialize(peak)
            db_peak.scan_id = scan_id
            out.append(db_peak)
        return out

    def __repr__(self):
        return "DB" + repr(self.convert())


def make_memory_fitted_peak(self):
    return MemoryFittedPeak(
        self.mz, self.intensity, self.signal_to_noise, -1, -1,
        self.full_width_at_half_max, (self.area if self.area is not None else 0.))


class FittedPeak(Base, PeakMixin):
    __tablename__ = "FittedPeak"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey(
        MSScan.id, ondelete='CASCADE'), index=True)

    convert = make_memory_fitted_peak

    @classmethod
    def serialize(cls, peak):
        return cls(
            mz=peak.mz, intensity=peak.intensity, signal_to_noise=peak.signal_to_noise,
            full_width_at_half_max=peak.full_width_at_half_max, area=peak.area)


def make_memory_deconvoluted_peak(self):
    return MemoryDeconvolutedPeak(
        self.neutral_mass, self.intensity, self.charge,
        self.signal_to_noise, -1, self.full_width_at_half_max,
        self.a_to_a2_ratio, self.most_abundant_mass, self.average_mass,
        self.score, Envelope(
            self.envelope), self.mz, None, self.chosen_for_msms,
        (self.area if self.area is not None else 0.))


class DeconvolutedPeak(Base, PeakMixin):
    __tablename__ = "DeconvolutedPeak"

    id = Column(Integer, primary_key=True, autoincrement=True)
    neutral_mass = Mass(False)
    average_mass = Mass(False)
    most_abundant_mass = Mass(False)

    charge = Column(Integer)

    score = Column(Numeric(12, 6, asdecimal=False))
    scan_id = Column(Integer, ForeignKey(
        MSScan.id, ondelete='CASCADE'), index=True)
    envelope = Column(MutableList.as_mutable(PickleType))
    a_to_a2_ratio = Column(Numeric(8, 7, asdecimal=False))
    chosen_for_msms = Column(Boolean)

    def __eq__(self, other):
        return (abs(self.neutral_mass - other.neutral_mass) < 1e-5) and (
            abs(self.intensity - other.intensity) < 1e-5)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.mz, self.intensity, self.charge))

    convert = make_memory_deconvoluted_peak

    @classmethod
    def serialize(cls, peak):
        return cls(
            mz=peak.mz, intensity=peak.intensity, signal_to_noise=peak.signal_to_noise,
            full_width_at_half_max=peak.full_width_at_half_max,
            neutral_mass=peak.neutral_mass, average_mass=peak.average_mass,
            most_abundant_mass=peak.most_abundant_mass, charge=peak.charge, score=peak.score,
            envelope=list(peak.envelope), a_to_a2_ratio=peak.a_to_a2_ratio,
            chosen_for_msms=peak.chosen_for_msms, area=(peak.area if peak.area is not None else 0.))


def serialize_scan_bunch(session, bunch, sample_run_id=None):
    precursor = bunch.precursor
    db_precursor = MSScan.serialize(precursor, sample_run_id=sample_run_id)
    session.add(db_precursor)
    db_products = [MSScan.serialize(p, sample_run_id=sample_run_id)
                   for p in bunch.products]
    session.add_all(db_products)
    session.flush()
    for scan, db_scan in zip(bunch.products, db_products):
        pi = scan.precursor_information
        db_pi = PrecursorInformation.serialize(
            pi, db_precursor, db_scan, sample_run_id)
        session.add(db_pi)
    session.flush()
    return db_precursor, db_products


def serialize_scan_bunch_bulk(session, bunch, sample_run_id, ms1_fitted=True, msn_fitted=True):
    precursor = bunch.precursor
    db_precursor = MSScan.serialize_bulk(
        precursor, sample_run_id, session, fitted=ms1_fitted)
    db_products = [MSScan.serialize_bulk(p, sample_run_id, session, fitted=msn_fitted)
                   for p in bunch.products]
    for scan, db_scan in zip(bunch.products, db_products):
        pi = scan.precursor_information
        db_pi = PrecursorInformation(
            precursor_id=db_precursor.id, product_id=db_scan.id,
            charge=pi.extracted_charge, intensity=pi.extracted_intensity,
            neutral_mass=pi.extracted_neutral_mass, sample_run_id=sample_run_id)
        session.add(db_pi)
    session.flush()
    return db_precursor, db_products
