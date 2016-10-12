from multiprocessing import Queue, Event
from glycan_profiling.serialize.hypothesis.peptide import Peptide, Protein

from .common import GlycopeptideHypothesisSerializerBase, PeptideGlycosylator, PeptideGlycosylatingProcess
from .proteomics import mzid_proteome


class MzIdentMLGlycopeptideHypothesisSerializer(GlycopeptideHypothesisSerializerBase):
    _display_name = "MzIdentML Glycopeptide Hypothesis Serializer"

    def __init__(self, mzid_path, connection, glycan_hypothesis_id, hypothesis_name=None,
                 target_proteins=None, max_glycosylation_events=1):
        if target_proteins is None:
            target_proteins = []
        GlycopeptideHypothesisSerializerBase.__init__(self, connection, hypothesis_name, glycan_hypothesis_id)
        self.mzid_path = mzid_path
        self.proteome = mzid_proteome.Proteome(
            mzid_path, self._original_connection, self.hypothesis_id, target_proteins=target_proteins)
        self.target_proteins = target_proteins
        self.max_glycosylation_events = max_glycosylation_events

    def retrieve_target_protein_ids(self):
        if len(self.target_proteins) == 0:
            return [
                i[0] for i in
                self.query(Protein.id).filter(
                    Protein.hypothesis_id == self.hypothesis_id).all()
            ]
        else:
            result = []
            for target in self.target_proteins:
                if isinstance(target, str):
                    match = self.query(Protein.id).filter(
                        Protein.name == target,
                        Protein.hypothesis_id == self.hypothesis_id).first()
                    if match:
                        result.append(match[0])
                    else:
                        self.log("Could not locate protein '%s'" % target)
                elif isinstance(target, int):
                    result.append(target)
            return result

    def peptide_ids(self):
        out = []
        for protein_id in self.retrieve_target_protein_ids():
            out.extend(i[0] for i in self.query(Peptide.id).filter(
                Peptide.protein_id == protein_id))
        return out

    def glycosylate_peptides(self):
        glycosylator = PeptideGlycosylator(self.session, self.hypothesis_id)
        acc = []
        i = 0
        for peptide_id in self.peptide_ids():
            peptide = self.query(Peptide).get(peptide_id)
            for glycopeptide in glycosylator.handle_peptide(peptide):
                acc.append(glycopeptide)
                i += 1
                if len(acc) > 100000:
                    self.session.add_all(acc)
                    self.session.commit()
                    acc = []
        self.session.add_all(acc)
        self.session.commit()

    def run(self):
        self.log("Loading Proteome")
        self.proteome.load()
        self.log("Combinating Glycans")
        self.combinate_glycans(self.max_glycosylation_events)
        self.log("Building Glycopeptides")
        self.glycosylate_peptides()
        self._count_produced_glycopeptides()
        self.log("Done")


class MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer(MzIdentMLGlycopeptideHypothesisSerializer):
    _display_name = "Multiple Process MzIdentML Glycopeptide Hypothesis Serializer"

    def __init__(self, mzid_path, connection, glycan_hypothesis_id, hypothesis_name=None,
                 target_proteins=None, max_glycosylation_events=1, n_processes=4):
        super(MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer, self).__init__(
            mzid_path, connection, glycan_hypothesis_id, hypothesis_name, target_proteins,
            max_glycosylation_events)
        self.n_processes = n_processes

    def glycosylate_peptides(self):
        input_queue = Queue(15)
        done_event = Event()
        processes = [
            PeptideGlycosylatingProcess(
                self._original_connection, self.hypothesis_id, input_queue,
                chunk_size=15000, done_event=done_event) for i in range(self.n_processes)
        ]
        peptide_ids = self.peptide_ids()
        i = 0
        chunk_size = 20
        for process in processes:
            input_queue.put(peptide_ids[i:(i + chunk_size)])
            i += chunk_size
            process.start()

        while i < len(peptide_ids):
            input_queue.put(peptide_ids[i:(i + chunk_size)])
            i += chunk_size
            self.log("... Dealt Peptides %d-%d" % (i - chunk_size, i))

        self.log("... All Peptides Dealt")
        done_event.set()
        for process in processes:
            process.join()
