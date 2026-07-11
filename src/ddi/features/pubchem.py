"""Drug name to SMILES resolution.

Resolution order:
  1. A DrugBank reference table (JSON) if one has been loaded via
     load_reference_table() — the fully-offline path built from the XML.
  2. A small built-in offline table for the curated reference drugs.
  3. The PubChem REST API (only if allow_network and the name is unresolved).
"""

import json
import os
from functools import lru_cache

# Populated by load_reference_table(); maps lowercase name -> {"smiles": ...}.
_REFERENCE_TABLE = {}


def load_reference_table(path):
    """Load a DrugBank-derived name->{smiles, cyp, ...} JSON produced by
    extract_drug_reference(). Returns the number of entries loaded.

    Synonyms are expanded into the in-memory index (aliasing the same entry
    object) rather than being duplicated on disk, so 'aspirin' resolves to the
    entry DrugBank files under 'acetylsalicylic acid'. A primary name always
    wins over an alias.
    """
    global _REFERENCE_TABLE
    if not path or not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        table = json.load(fh)

    n_primary = len(table)
    for name, entry in list(table.items()):
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("aliases") or []:
            if alias and alias not in table:
                table[alias] = entry

    _REFERENCE_TABLE = table
    resolve_smiles.cache_clear()
    return n_primary


def reference_table():
    return _REFERENCE_TABLE

OFFLINE_SMILES = {
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "fluconazole": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
    "simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
    "clarithromycin": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(OC)CC(C)C(=O)C(C)C(O)(C)C1C",
    "ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "theophylline": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    "codeine": "CN1CCC23C=CC(O)C(OC)C2Oc2c(O)ccc(C1)c23",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "paroxetine": "Fc1ccc(cc1)C1CCNCC1COc1ccc2OCOc2c1",
    "metoprolol": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
    "omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "clopidogrel": "COC(=O)C(c1ccccc1Cl)N1CCc2c(C1)ccs2",
    "ritonavir": "CC(C)c1nc(CN(C)C(=O)NC(C(=O)NC(Cc2ccccc2)CC(O)C(Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1",
    "sildenafil": "CCCc1nn(C)c2c1nc([nH]c2=O)c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "amiodarone": "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
    "haloperidol": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
    "amoxicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "trimethoprim": "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",
    "sulfamethoxazole": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    "lisinopril": "NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",
    "spironolactone": "CC(=O)SC1CC2C3CCC4=CC(=O)CCC4(C)C3CCC2(C)C11CCC(=O)O1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


@lru_cache(maxsize=131072)  # must exceed the ~55k reference keys to be useful
def resolve_smiles(drug_name: str, allow_network: bool = True):
    key = drug_name.strip().lower()
    entry = _REFERENCE_TABLE.get(key)
    if entry and entry.get("smiles"):
        return entry["smiles"]
    if key in OFFLINE_SMILES:
        return OFFLINE_SMILES[key]
    if not allow_network:
        return None
    try:
        import httpx
        from urllib.parse import quote
        # Percent-encode: the raw name lands in the URL path, so an unescaped
        # '/', '?', or '..' would rewrite the request.
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{quote(drug_name, safe='')}/property/CanonicalSMILES/JSON")
        r = httpx.get(url, timeout=5.0)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"]
            if props:
                return props[0]["CanonicalSMILES"]
    except Exception:
        return None
    return None
