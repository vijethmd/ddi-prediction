"""NLP keyword classifier calibrated to real DrugBank 6.0 templated phrasing."""

from .. import CLASS_INDEX

SERIOUS_EFFECTS = [
    "bleeding", "hemorrhage", "haemorrhage", "cns depression",
    "respiratory depression", "qt", "torsade", "arrhythmia",
    "serotonin syndrome", "hyperkalemia", "hyperkalaemia",
    "hypoglycemia", "hypoglycaemia", "rhabdomyolysis", "seizure",
    "neuroleptic malignant", "hypertensive crisis", "cardiotoxicity",
    "hepatotoxicity", "nephrotoxicity", "myelosuppression",
    "bone marrow", "lactic acidosis", "methemoglobinemia",
    "hypotension", "bradycardia", "cardiac", "pancreatitis",
    "hyperthermia", "toxicity",
]
SEVERE_EXPLICIT = [
    "contraindicated", "avoid concurrent", "avoid the combination",
    "should not be co-administered", "must not be", "life-threatening",
    "fatal", "potentially fatal",
]

# Direct efficacy verbs. DrugBank 6.0's templated text uses the
# "therapeutic efficacy ... increased/decreased" form instead, so these fire on
# free-text sources only; they match no row of the current corpus. The word
# "antagonist" is deliberately excluded -- it names a drug class, not an effect.
SYNERGISTIC_CUES = ["potentiate", "enhances the effect", "synergistic"]
ANTAGONISTIC_CUES = ["reduces efficacy", "reduces the efficacy",
                     "decreases efficacy", "counteract"]

def classify_description(description: str) -> int:
    if not description:
        return CLASS_INDEX["No Interaction"]
    t = description.lower()
    if any(k in t for k in SEVERE_EXPLICIT):
        return CLASS_INDEX["Severe"]
    if "therapeutic efficacy" in t:
        if "increased" in t:
            return CLASS_INDEX["Synergistic"]
        if "decreased" in t:
            return CLASS_INDEX["Antagonistic"]
    if any(k in t for k in ANTAGONISTIC_CUES):
        return CLASS_INDEX["Antagonistic"]
    if any(k in t for k in SYNERGISTIC_CUES):
        return CLASS_INDEX["Synergistic"]
    if "risk or severity" in t or "increase the risk" in t:
        if any(e in t for e in SERIOUS_EFFECTS):
            return CLASS_INDEX["Severe"]
        return CLASS_INDEX["Moderate"]
    if any(e in t for e in SERIOUS_EFFECTS) and "increas" in t:
        return CLASS_INDEX["Severe"]
    moderate_cues = [
        "excretion rate", "serum level", "serum concentration",
        "metabolism", "absorption", "may increase", "may decrease",
        "activities of", "activity of", "monitor", "dose",
    ]
    if any(k in t for k in moderate_cues):
        return CLASS_INDEX["Moderate"]
    return CLASS_INDEX["No Interaction"]

def label_dataframe(df, desc_col="description", out_col="severity"):
    df = df.copy()
    df[out_col] = df[desc_col].fillna("").apply(classify_description)
    return df

def label_stats(labels):
    from collections import Counter
    from .. import CLASS_NAMES
    counts = Counter(int(x) for x in labels)
    total = sum(counts.values()) or 1
    return {CLASS_NAMES[k]: {"count": counts.get(k, 0),
            "pct": round(100 * counts.get(k, 0) / total, 2)}
            for k in sorted(CLASS_NAMES)}
