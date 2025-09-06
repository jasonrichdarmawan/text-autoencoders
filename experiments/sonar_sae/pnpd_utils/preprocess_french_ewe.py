def preprocess_french_ewe(directory: str):
    """
    |- french_ewe.ewe
    |- french_ewe.fra
    """
    data = []
    with open(f"{directory}/french_ewe.ewe", "r", encoding="utf-8") as f_ewe, open(
        f"{directory}/french_ewe.fra", "r", encoding="utf-8"
    ) as f_fra:
        for line_ewe, line_fra in zip(f_ewe, f_fra):
            line_ewe = line_ewe.strip()
            line_fra = line_fra.strip()
            if len(line_ewe) == 0 or len(line_fra) == 0:
                continue
            data.append(
                {
                    "text_1": line_ewe,
                    "lang_1": "ewe_Latn",
                    "text_2": line_fra,
                    "lang_2": "fra_Latn",
                }
            )
    return data
