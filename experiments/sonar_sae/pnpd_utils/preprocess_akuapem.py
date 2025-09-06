def preprocess_akuapem(directory: str):
    """
    |- akuapem.aka
    |- akuapem.eng
    """
    data = []
    with open(f"{directory}/akuapem.aka", "r", encoding="utf-8") as f_aka, open(
        f"{directory}/akuapem.eng", "r", encoding="utf-8"
    ) as f_eng:
        for line_aka, line_eng in zip(f_aka, f_eng):
            line_aka = line_aka.strip()
            line_eng = line_eng.strip()
            if len(line_aka) == 0 or len(line_eng) == 0:
                continue
            data.append(
                {
                    "text_1": line_aka,
                    "lang_1": "aka_Latn",
                    "text_2": line_eng,
                    "lang_2": "eng_Latn",
                }
            )

    return data
