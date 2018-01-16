from .chromatogram import (get_chromatogram, mask_subsequence)
from .index import ChromatogramFilter


class AdductTreePruner(object):
    def __init__(self, solutions, score_margin=2.5, ratio_threshold=1.5, trivial_abundance_delta_ratio=0.01):
        self.solutions = solutions
        self.score_margin = score_margin
        self.ratio_threshold = ratio_threshold
        self.trivial_abundance_delta_ratio = trivial_abundance_delta_ratio

        self.key_map = self.solutions._build_key_map()
        self.updated = set()

    def check_close_score_abundance_ratio(self, case, owner_item):
        component_signal = case.total_signal
        complement_signal = owner_item.total_signal - component_signal
        signal_ratio = complement_signal / component_signal
        # The owner is more abundant than used-as-adduct-case
        return (signal_ratio < self.ratio_threshold)

    def create_masked_chromatogram(self, owner_item, case):
        new_masked = mask_subsequence(get_chromatogram(owner_item), get_chromatogram(case))
        new_masked.created_at = "prune_bad_adduct_branches"
        new_masked.score = owner_item.score
        return new_masked

    def handle_chromatogram(self, case):
        if case.used_as_adduct:
            keepers = []
            for owning_key, adduct in case.used_as_adduct:
                owner = self.key_map.get(owning_key)
                if owner is None:
                    continue
                owner_item = owner.find_overlap(case)
                if owner_item is None:
                    continue
                scores_close = abs(case.score - owner_item.score) < self.score_margin
                adducted_case_stronger = case.score > owner_item.score
                # If the owning group is lower scoring, but the scores are close
                if adducted_case_stronger and scores_close:
                    adducted_case_stronger = self.check_close_score_abundance_ratio(case, owner_item)
                # If the scores are close, but the owning group is less abundant,
                # e.g. more mass shift groups or mass accuracy prevents propagation
                # of mass shifts
                elif scores_close and (owner_item.total_signal / case.total_signal) < 1:
                    adducted_case_stronger = True
                independent = owner_item.total_signal - case.total_signal
                if (independent / case.total_signal) < self.trivial_abundance_delta_ratio:
                    adducted_case_stronger = True
                if adducted_case_stronger:
                    new_masked = self.create_masked_chromatogram(owner_item, case)
                    if len(new_masked) != 0:
                        owner.replace(owner_item, new_masked)
                    self.updated.add(owning_key)
                else:
                    keepers.append((owning_key, adduct))
            case.chromatogram.used_as_adduct = keepers

    def prune_branches(self):
        for case in self.solutions:
            self.handle_chromatogram(case)
        out = [s.chromatogram for k in (set(self.key_map) - self.updated) for s in self.key_map[k]]
        out.extend(s for k in self.updated for s in self.key_map[k])
        out = ChromatogramFilter(out)
        return out

    @classmethod
    def prune_bad_adduct_branches(cls, solutions, score_margin=2.5, ratio_threshold=1.5,
                                  trivial_abundance_delta_ratio=0.01):
        inst = cls(solutions, score_margin, ratio_threshold, trivial_abundance_delta_ratio)
        return inst.prune_branches()


prune_bad_adduct_branches = AdductTreePruner.prune_bad_adduct_branches
