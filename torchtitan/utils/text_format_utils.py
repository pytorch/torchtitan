# Adapted from https://github.com/YerevaNN/ChemLactica/blob/main/chemlactica/utils/text_format_utils.py
# All rights reserved

SPECIAL_TAGS = {
    "SMILES": {"start": "[START_SMILES]", "end": "[END_SMILES]"},
    "synonym": {"start": "[SYNONYM]", "end": "[/SYNONYM]"},
    "RELATED": {"start": "[RELATED]", "end": "[/RELATED]"},
    "similarity": {"start": "[SIMILAR]", "end": "[/SIMILAR]", "type": float},
    "PROPERTY": {"start": "[PROPERTY]", "end": "[/PROPERTY]"},
    "SAS": {"start": "[SAS]", "end": "[/SAS]", "type": float},
    "WEIGHT": {"start": "[WEIGHT]", "end": "[/WEIGHT]", "type": float},
    "TPSA": {"start": "[TPSA]", "end": "[/TPSA]", "type": float},
    "CLOGP": {"start": "[CLOGP]", "end": "[/CLOGP]", "type": float},
    "QED": {"start": "[QED]", "end": "[/QED]", "type": float},
    "NUMHDONORS": {"start": "[NUMHDONORS]", "end": "[/NUMHDONORS]"},
    "NUMHACCEPTORS": {"start": "[NUMHACCEPTORS]", "end": "[/NUMHACCEPTORS]"},
    "NUMHETEROATOMS": {"start": "[NUMHETEROATOMS]", "end": "[/NUMHETEROATOMS]"},
    "NUMROTATABLEBONDS": {
        "start": "[NUMROTATABLEBONDS]",
        "end": "[/NUMROTATABLEBONDS]",
    },
    "NOCOUNT": {"start": "[NOCOUNT]", "end": "[/NOCOUNT]"},
    "NHOHCOUNT": {"start": "[NHOHCOUNT]", "end": "[/NHOHCOUNT]"},
    "RINGCOUNT": {"start": "[RINGCOUNT]", "end": "[/RINGCOUNT]"},
    "HEAVYATOMCOUNT": {"start": "[HEAVYATOMCOUNT]", "end": "[/HEAVYATOMCOUNT]"},
    "FRACTIONCSP3": {
        "start": "[FRACTIONCSP3]",
        "end": "[/FRACTIONCSP3]",
        "type": float,
    },
    "NUMAROMATICRINGS": {
        "start": "[NUMAROMATICRINGS]",
        "end": "[/NUMAROMATICRINGS]",
    },
    "NUMSATURATEDRINGS": {
        "start": "[NUMSATURATEDRINGS]",
        "end": "[/NUMSATURATEDRINGS]",
    },
    "NUMAROMATICHETEROCYCLES": {
        "start": "[NUMAROMATICHETEROCYCLES]",
        "end": "[/NUMAROMATICHETEROCYCLES]",
    },
    "NUMAROMATICCARBOCYCLES": {
        "start": "[NUMAROMATICCARBOCYCLES]",
        "end": "[/NUMAROMATICCARBOCYCLES]",
    },
    "NUMSATURATEDHETEROCYCLES": {
        "start": "[NUMSATURATEDHETEROCYCLES]",
        "end": "[/NUMSATURATEDHETEROCYCLES]",
    },
    "NUMSATURATEDCARBOCYCLES": {
        "start": "[NUMSATURATEDCARBOCYCLES]",
        "end": "[/NUMSATURATEDCARBOCYCLES]",
    },
    "NUMALIPHATICRINGS": {
        "start": "[NUMALIPHATICRINGS]",
        "end": "[/NUMALIPHATICRINGS]",
    },
    "NUMALIPHATICHETEROCYCLES": {
        "start": "[NUMALIPHATICHETEROCYCLES]",
        "end": "[/NUMALIPHATICHETEROCYCLES]",
    },
    "NUMALIPHATICCARBOCYCLES": {
        "start": "[NUMALIPHATICCARBOCYCLES]",
        "end": "[/NUMALIPHATICCARBOCYCLES]",
    },
    "IUPAC": {"start": "[IUPAC]", "end": "[/IUPAC]"},
    "VAR_NAME": {"start": "[VAR_NAME]", "end": "[/VAR_NAME]"},
    "VAR_DESC": {"start": "[VAR_DESC]", "end": "[/VAR_DESC]"},
    "VAR_VAL": {"start": "[VAR_VAL]", "end": "[/VAR_VAL]"},
    "ASSAY_NAME": {"start": "[ASSAY_NAME]", "end": "[/ASSAY_NAME]"},
    "ASSAY_DESC": {"start": "[ASSAY_DESC]", "end": "[/ASSAY_DESC]"},
    "formula": {"start": "[FORMULA]", "end": "[/FORMULA]"},
}


def delete_empty_tags(compound_json):
    for k, v in list(compound_json.items()):
        if v == [] or v == "":
            del compound_json[k]
    return compound_json


def generate_formatted_string(compound_json, rng):
    key_value_pairs = []
    key = "SMILES"
    value = compound_json.get(key, "")
    if rng.integers(2) == 0:
        if value:
            key_value_pairs.append(format_key_value(key, value, rng))
            del compound_json[key]
    keys = list(compound_json.keys())
    rng.shuffle(keys)

    for key in keys:
        key_value_pairs.append(format_key_value(key, compound_json[key], rng))
    compound_formatted_string = (
        "".join(key_value_pairs)
    )
    return compound_formatted_string


def format_key_value(key, value, rng):
    if key == "CID":
        return ""
    formatted_string = ""
    if key == "related":
        if len(value) > 10:
            # value = random.sample(value, 5)
            value = rng.choice(value, size=10, replace=False, shuffle=False)
        for pair in value:
            rounded_sim = "{:.2f}".format(float(pair["similarity"]))
            formatted_string += f"{SPECIAL_TAGS['similarity']['start']}{pair['SMILES']} {rounded_sim}{SPECIAL_TAGS['similarity']['end']}"  # noqa
    elif key == "experimental":
        for pair in value:
            formatted_string += f"[PROPERTY]{pair['PROPERTY_NAME']} {pair['PROPERTY_VALUE']}[/PROPERTY]"  # noqa
    elif key == "synonyms":
        for val in value:
            formatted_string += f"{SPECIAL_TAGS['synonym']['start']}{val['name']}{SPECIAL_TAGS['synonym']['end']}"  # noqa
    else:
        try:
            if SPECIAL_TAGS[key].get("type") is float:
                value = "{:.2f}".format(float(value))
                assert len(value.split(".")[-1]) == 2
            start = SPECIAL_TAGS[key]["start"]
            end = SPECIAL_TAGS[key]["end"]
        except Exception as e:
            print(e)
            print("Failed to parse: ", key, value)
            start = value = end = ""
        return f"{start}{value}{end}"

    return formatted_string