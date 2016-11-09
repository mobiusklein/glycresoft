from glypy.structure.substituent import register as register_substituent_rule
from glypy.composition import formula


substituent_schema = {
    "name": str,
    "composition": str,
    "can_nh_derivatize": bool,
    "is_nh_derivatizable": bool,
    "attachment_composition": str
}


def serialize_substituent_rule(substituent):
    return {
        "name": substituent.name,
        "composition": formula(substituent.composition),
        "can_nh_derivatize": substituent.can_nh_derivatize,
        "is_nh_derivatizable": substituent.is_nh_derivatizable,
        "attachment_composition": formula(substituent.attachment_composition)
    }


def load_substituent_rule(rule_dict):
    return register_substituent_rule(
        rule_dict['name'], rule_dict['composition'],
        rule_dict['can_nh_derivatize'], rule_dict['is_nh_derivatizable'],
        rule_dict['attachment_composition'])


def save_substituent_rule(substituent):
    return serialize_substituent_rule(substituent)
