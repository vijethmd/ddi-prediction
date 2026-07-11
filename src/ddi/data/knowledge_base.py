"""CYP450 enzyme knowledge base and pharmacodynamic interaction rules.

This is a curated reference subset used for (a) engineering CYP conflict features
and (b) the rule-based prototype fallback. In Phase 2 the full profile table is
loaded from DrugBank; this module holds the high-value pairs used in testing and
the deployed demo.
"""

# Per-drug CYP450 role profiles: for each drug, which isoforms it inhibits,
# induces, or is a substrate of. Names are lowercase for lookup robustness.
CYP_PROFILES = {
    "fluconazole": {"inhibits": ["CYP2C9", "CYP2C19", "CYP3A4"], "substrate": []},
    "warfarin": {"inhibits": [], "substrate": ["CYP2C9", "CYP3A4", "CYP1A2"]},
    "clarithromycin": {"inhibits": ["CYP3A4"], "substrate": ["CYP3A4"]},
    "simvastatin": {"inhibits": [], "substrate": ["CYP3A4"]},
    "ciprofloxacin": {"inhibits": ["CYP1A2"], "substrate": []},
    "theophylline": {"inhibits": [], "substrate": ["CYP1A2"]},
    "fluoxetine": {"inhibits": ["CYP2D6", "CYP2C19"], "substrate": ["CYP2D6"]},
    "codeine": {"inhibits": [], "substrate": ["CYP2D6"]},
    "paroxetine": {"inhibits": ["CYP2D6"], "substrate": ["CYP2D6"]},
    "metoprolol": {"inhibits": [], "substrate": ["CYP2D6"]},
    "omeprazole": {"inhibits": ["CYP2C19"], "substrate": ["CYP2C19"]},
    "clopidogrel": {"inhibits": [], "substrate": ["CYP2C19", "CYP3A4"]},
    "ritonavir": {"inhibits": ["CYP3A4"], "substrate": ["CYP3A4"]},
    "sildenafil": {"inhibits": [], "substrate": ["CYP3A4"]},
    "amiodarone": {"inhibits": ["CYP2C9", "CYP2D6", "CYP3A4"], "substrate": ["CYP3A4"]},
    "haloperidol": {"inhibits": ["CYP2D6"], "substrate": ["CYP2D6", "CYP3A4"]},
    "amoxicillin": {"inhibits": [], "substrate": []},
    "metformin": {"inhibits": [], "substrate": []},
    "trimethoprim": {"inhibits": ["CYP2C8"], "substrate": []},
    "sulfamethoxazole": {"inhibits": ["CYP2C9"], "substrate": ["CYP2C9"]},
    "lisinopril": {"inhibits": [], "substrate": []},
    "spironolactone": {"inhibits": [], "substrate": ["CYP3A4"]},
}

# Curated reference pairs with ground-truth class and mechanism. Used for
# integration tests and the prototype's known-pair fast path.
REFERENCE_PAIRS = {
    ("warfarin", "fluconazole"): {
        "label": 2, "cyp": "CYP2C9",
        "mechanism": "Fluconazole inhibits CYP2C9, which metabolises warfarin. "
                     "Warfarin levels rise, sharply increasing bleeding risk.",
        "action": "Avoid or reduce warfarin dose with close INR monitoring.",
    },
    ("simvastatin", "clarithromycin"): {
        "label": 2, "cyp": "CYP3A4",
        "mechanism": "Clarithromycin strongly inhibits CYP3A4, which clears "
                     "simvastatin. Statin levels rise, risking rhabdomyolysis.",
        "action": "Contraindicated. Suspend simvastatin during therapy.",
    },
    ("theophylline", "ciprofloxacin"): {
        "label": 2, "cyp": "CYP1A2",
        "mechanism": "Ciprofloxacin inhibits CYP1A2, which metabolises "
                     "theophylline. Theophylline accumulates to toxic levels.",
        "action": "Avoid combination or monitor theophylline levels closely.",
    },
    ("codeine", "fluoxetine"): {
        "label": 2, "cyp": "CYP2D6",
        "mechanism": "Fluoxetine inhibits CYP2D6, blocking conversion of codeine "
                     "to morphine — analgesia is lost and serotonergic risk rises.",
        "action": "Choose an alternative analgesic.",
    },
    ("warfarin", "aspirin"): {
        "label": 2, "cyp": None,
        "mechanism": "Additive pharmacodynamic effect: both impair haemostasis "
                     "(anticoagulation plus antiplatelet), compounding bleeding risk.",
        "action": "Avoid unless a specific indication justifies dual therapy.",
    },
    ("haloperidol", "amiodarone"): {
        "label": 2, "cyp": None,
        "mechanism": "Both prolong the QT interval; combined use raises the risk "
                     "of torsades de pointes.",
        "action": "Avoid combination; monitor ECG if unavoidable.",
    },
    ("sildenafil", "ritonavir"): {
        "label": 2, "cyp": "CYP3A4",
        "mechanism": "Ritonavir strongly inhibits CYP3A4, which clears sildenafil. "
                     "Sildenafil levels rise, risking severe hypotension.",
        "action": "Reduce sildenafil dose substantially; avoid high doses.",
    },
    ("clopidogrel", "omeprazole"): {
        "label": 1, "cyp": "CYP2C19",
        "mechanism": "Omeprazole inhibits CYP2C19, which activates clopidogrel. "
                     "Antiplatelet efficacy may be reduced.",
        "action": "Prefer pantoprazole; monitor for reduced antiplatelet effect.",
    },
    ("lisinopril", "spironolactone"): {
        "label": 1, "cyp": None,
        "mechanism": "Both raise serum potassium; combined use risks hyperkalaemia.",
        "action": "Monitor serum potassium and renal function.",
    },
    ("metoprolol", "paroxetine"): {
        "label": 1, "cyp": "CYP2D6",
        "mechanism": "Paroxetine inhibits CYP2D6, which metabolises metoprolol. "
                     "Metoprolol levels rise, enhancing beta-blockade.",
        "action": "Monitor heart rate and blood pressure; adjust dose.",
    },
    ("trimethoprim", "sulfamethoxazole"): {
        "label": 3, "cyp": None,
        "mechanism": "Sequential blockade of folate synthesis: trimethoprim and "
                     "sulfamethoxazole act on consecutive steps, giving synergy.",
        "action": "Intended therapeutic combination (co-trimoxazole).",
    },
    ("amoxicillin", "metformin"): {
        "label": 0, "cyp": None,
        "mechanism": "No clinically significant interaction; distinct clearance "
                     "pathways and no shared pharmacodynamic target.",
        "action": "No action required.",
    },
}


# Optional override populated from the DrugBank reference JSON. When loaded, CYP
# lookups consult this first, giving full-catalogue coverage beyond the curated
# CYP_PROFILES above.
_REFERENCE_CYP = {}

# Populated alongside _REFERENCE_CYP: lowercase drug name -> set of UniProt
# accessions. Drives the target_overlap_count feature.
_REFERENCE_TARGETS = {}

_EMPTY_PROFILE = {"inhibits": [], "substrate": [], "induces": [], "strength": {}}

_ROLES = ("inhibits", "substrate", "induces")


def _coerce(profile):
    """Normalise any profile dict to the full role set.

    The curated CYP_PROFILES above predate the 'induces'/'strength' fields that
    extract_drug_reference() now emits, so fill the gaps rather than KeyError.
    """
    if not profile:
        return dict(_EMPTY_PROFILE)
    return {
        "inhibits": list(profile.get("inhibits") or []),
        "substrate": list(profile.get("substrate") or []),
        "induces": list(profile.get("induces") or []),
        "strength": dict(profile.get("strength") or {}),
    }


def load_reference_cyp(reference_table):
    """Populate the CYP and target overrides from a DrugBank reference table
    (name -> entry with 'cyp' and 'targets' blocks). Returns the number of drugs
    with any CYP data."""
    global _REFERENCE_CYP, _REFERENCE_TARGETS
    _REFERENCE_CYP = {}
    _REFERENCE_TARGETS = {}
    # Synonyms alias the same entry object, so count distinct drugs, not keys.
    distinct = set()
    for name, entry in reference_table.items():
        if not isinstance(entry, dict):
            continue
        cyp = entry.get("cyp")
        if cyp and any(cyp.get(role) for role in _ROLES):
            _REFERENCE_CYP[name] = _coerce(cyp)
            distinct.add(entry.get("drugbank_id") or name)
        targets = entry.get("targets")
        if targets:
            _REFERENCE_TARGETS[name] = set(targets)
    return len(distinct)


def target_overlap(drug_a: str, drug_b: str) -> int:
    """Number of protein targets the two drugs share.

    Shared targets are the pharmacodynamic counterpart to a CYP conflict: two
    drugs acting on the same protein produce additive, synergistic, or
    antagonistic effects without either changing the other's concentration.
    Returns 0 when either drug has no target annotation.
    """
    ta = _REFERENCE_TARGETS.get(normalise(drug_a))
    tb = _REFERENCE_TARGETS.get(normalise(drug_b))
    if not ta or not tb:
        return 0
    return len(ta & tb)


def _profile(name):
    """Return the CYP profile for a drug, preferring the loaded reference
    table, then the curated CYP_PROFILES, else an empty profile."""
    key = name.strip().lower()
    if key in _REFERENCE_CYP:
        return _REFERENCE_CYP[key]
    return _coerce(CYP_PROFILES.get(key))


def normalise(name: str) -> str:
    return name.strip().lower()


def lookup_reference(drug_a: str, drug_b: str):
    a, b = normalise(drug_a), normalise(drug_b)
    if (a, b) in REFERENCE_PAIRS:
        return REFERENCE_PAIRS[(a, b)]
    if (b, a) in REFERENCE_PAIRS:
        return REFERENCE_PAIRS[(b, a)]
    return None


def cyp_conflicts(drug_a: str, drug_b: str):
    """Return list of (isoform, inhibitor, substrate) conflicts where one drug
    inhibits an isoform the other is a substrate of."""
    a, b = normalise(drug_a), normalise(drug_b)
    pa = _profile(a)
    pb = _profile(b)
    if not any(pa[r] for r in _ROLES) or not any(pb[r] for r in _ROLES):
        return []
    conflicts = []
    for iso in set(pa["inhibits"]) & set(pb["substrate"]):
        conflicts.append((iso, a, b))
    for iso in set(pb["inhibits"]) & set(pa["substrate"]):
        conflicts.append((iso, b, a))
    return conflicts
