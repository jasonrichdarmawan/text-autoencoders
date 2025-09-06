def preprocess_nynorsk_memories(directory: str):
    """
    |- nynorsk_memories.nno
    |- nynorsk_memories.nob
    """
    data = []
    with open(
        f"{directory}/nynorsk_memories.nno", "r", encoding="utf-8"
    ) as f_nno, open(
        f"{directory}/nynorsk_memories.nob", "r", encoding="utf-8"
    ) as f_nob:
        for line_nno, line_nob in zip(f_nno, f_nob):
            line_nno = line_nno.strip()
            line_nob = line_nob.strip()
            if len(line_nno) == 0 or len(line_nob) == 0:
                continue
            data.append(
                {
                    "text_1": line_nno,
                    "lang_1": "nno_Latn",
                    "text_2": line_nob,
                    "lang_2": "nob_Latn",
                }
            )

    return data
