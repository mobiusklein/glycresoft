from glycopeptidepy.structure.modification import (
    ModificationTarget, ModificationRule, Modification,
    ModificationCategory, SequenceLocation)


modification_rule_schema = {
    "composition": dict,
    "mono_mass": float,
    "full_name": str,
    "title": str,
    "specificity": list,
}


modification_specificity_schema = {
    "position": str,
    "site": str
}


position_map = {
    SequenceLocation.anywhere: "Anywhere",
    SequenceLocation.n_term: "Any N-term",
    SequenceLocation.c_term: "Any C-term",
}


def serialize_modification_target(target):
    payloads = []
    if not target.amino_acid_targets:
        payloads.append({
            "site": None,
            "position": position_map[target.position_modifier]
        })
        return payloads
    else:
        for aa in target.amino_acid_targets:
            payloads.append({
                "site": str(aa),
                "position": position_map[target.position_modifier]
            })
        return payloads


def serialize_modification_rule(rule):
    payload = dict()
    payload['mono_mass'] = rule.mass
    payload['full_name'] = rule.name
    payload['title'] = rule.title
    payload['composition'] = dict(rule.composition)
    payload['specificity'] = [s for t in rule.targets for s in serialize_modification_target(t)]
    return payload


def load_modification_rule(rule_dict):
    rule = ModificationRule.from_unimod(rule_dict)
    Modification.register_new_rule(rule)
    return rule


def save_modification_rule(rule):
    return serialize_modification_rule(rule)
