def preprocess_french_fongbe(directory: str):
    """
    |- french_fongbe.fon
    |- french_fongbe.fra
    """
    data = []
    with open(f"{directory}/french_fongbe.fon", "r", encoding="utf-8") as f_fon, open(
        f"{directory}/french_fongbe.fra", "r", encoding="utf-8"
    ) as f_fra:
        for line_fon, line_fra in zip(f_fon, f_fra):
            line_fon = line_fon.strip()
            line_fra = line_fra.strip()
            if len(line_fon) == 0 or len(line_fra) == 0:
                continue
            data.append(
                {
                    "text_1": line_fon,
                    "lang_1": "fon_Latn",
                    "text_2": line_fra,
                    "lang_2": "fra_Latn",
                }
            )
    return data
