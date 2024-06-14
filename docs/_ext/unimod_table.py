from glycopeptidepy import Modification
from rst_table import as_rest_table

rows = [["UNIMOD", "Name", "Mass", "Other Names", "Targets"]]

def unimod_name(rule):
    names = list(filter(lambda x: x.startswith("UNIMOD"), rule.names))
    if names:
        return names[0]

rules = Modification._table.rules()
unimod_rules = sorted(filter(unimod_name, rules), key=lambda x: int(unimod_name(x).split(":")[1]))

for rule in unimod_rules:
    row = [
        unimod_name(rule),
        rule.preferred_name,
        rule.mass,
        ', '.join(rule.names - {unimod_name(rule), rule.preferred_name}),
        ', '.join(map(lambda x: x.replace(rule.name, '')[2:-1], rule.as_spec_strings()))
    ]
    rows.append(row)

rendered_table = (as_rest_table(rows))