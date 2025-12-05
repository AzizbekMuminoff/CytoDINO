LINEAGE_GROUPS = {
    "granulocytic": ["BLA", "PMO", "MYB", "MMZ", "NGB", "NGS"],
    "erythroid": ["PEB", "EBO"],
    "lymphoid": ["LYI", "LYT", "PLM"],
    "monocytic": ["MON"],
    "eosinophil": ["EOS", "ABE"],
    "basophil": ["BAS"],
    "pathological": ["FGC", "HAC", "KSC"],
    "artifact": ["ART", "NIF", "OTH"],
}

SEQUENTIAL_LINEAGES = {
    "granulocytic": ["BLA", "PMO", "MYB", "MMZ", "NGB", "NGS"],
    "erythroid": ["PEB", "EBO"],
}

CRITICAL_CLASSES = ["BLA", "MYB", "PMO", "FGC", "HAC", "KSC", "ABE", "LYI"] 
SAFE_CLASSES = ["NGS", "NGB", "LYT", "MON", "EOS", "BAS", "PLM", "EBO"]