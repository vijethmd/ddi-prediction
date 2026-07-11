"""Molecular feature engineering for drug pairs.

Computes 2,423 features per pair: ECFP4 Morgan fingerprints (2 x 1024), MACCS
keys (2 x 167), seven physicochemical descriptors per drug, a 25-dimensional
CYP450 block, Tanimoto similarity, and shared-target count.

RDKit is a hard requirement. It used to be optional, with structural features
degrading to zeros when it was missing -- but the model is trained on real
fingerprints, so feeding it an all-zero vector produces a confident, silent,
wrong answer rather than an error. Missing RDKit is now fatal, and an
unresolved SMILES raises UnresolvedStructure so the caller decides what to do.
"""

import numpy as np

from ..data.knowledge_base import (
    cyp_conflicts, normalise, _profile, target_overlap,
)
from .. import CYP_ISOFORMS

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    RDKIT_AVAILABLE = False


class UnresolvedStructure(ValueError):
    """Raised when a drug's SMILES cannot be resolved to a molecule.

    Callers must handle this explicitly: training drops the pair, serving
    declines to make a model prediction. Never substitute zeros.
    """

ECFP_BITS = 1024
MACCS_BITS = 167  # RDKit emits 167 (index 0 unused)
PHYSCHEM = ["MolWt", "MolLogP", "TPSA", "NumHDonors",
            "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings"]


def feature_names():
    """Return the ordered list of all feature names."""
    names = []
    for tag in ("a", "b"):
        names += [f"ecfp4_{tag}_{i}" for i in range(ECFP_BITS)]
    for tag in ("a", "b"):
        names += [f"maccs_{tag}_{i}" for i in range(MACCS_BITS)]
    for tag in ("a", "b"):
        names += [f"physchem_{tag}_{p}" for p in PHYSCHEM]
    for iso in CYP_ISOFORMS:
        names += [f"cyp_{iso}_a_inhibits", f"cyp_{iso}_a_substrate",
                  f"cyp_{iso}_b_inhibits", f"cyp_{iso}_b_substrate",
                  f"cyp_{iso}_conflict"]
    names += ["tanimoto_sim", "target_overlap_count"]
    return names


N_FEATURES = len(feature_names())


def _ecfp(mol):
    if mol is None:
        return np.zeros(ECFP_BITS)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=ECFP_BITS)
    arr = np.zeros((ECFP_BITS,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _maccs(mol):
    if mol is None:
        return np.zeros(MACCS_BITS)
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((MACCS_BITS,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _physchem(mol):
    if mol is None:
        return np.zeros(len(PHYSCHEM))
    fns = {
        "MolWt": Descriptors.MolWt,
        "MolLogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumAromaticRings": Descriptors.NumAromaticRings,
    }
    return np.array([fns[p](mol) for p in PHYSCHEM], dtype=np.float64)


def _cyp_block(drug_a, drug_b):
    a, b = normalise(drug_a), normalise(drug_b)
    pa = _profile(a)
    pb = _profile(b)
    conflict_isos = {iso for iso, _, _ in cyp_conflicts(drug_a, drug_b)}
    block = []
    for iso in CYP_ISOFORMS:
        block += [
            float(iso in pa["inhibits"]),
            float(iso in pa["substrate"]),
            float(iso in pb["inhibits"]),
            float(iso in pb["substrate"]),
            float(iso in conflict_isos),
        ]
    return np.array(block, dtype=np.float64)


def _tanimoto(mol_a, mol_b):
    if mol_a is None or mol_b is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, ECFP_BITS)
    fb = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, ECFP_BITS)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def _canonical_order(smiles_a, smiles_b, drug_a, drug_b):
    """Order the pair deterministically so f(A,B) == f(B,A).

    A drug interaction is symmetric, but the feature layout is not: swapping the
    arguments swaps the 'a' and 'b' fingerprint blocks and yields a different
    vector, hence a different probability. Near the Severe decision threshold
    that flips the served class purely on argument order. Sorting on the drug
    name (falling back to SMILES) removes the degree of freedom.
    """
    key_a = (drug_a or "").strip().lower() or (smiles_a or "")
    key_b = (drug_b or "").strip().lower() or (smiles_b or "")
    if key_a > key_b:
        return smiles_b, smiles_a, drug_b, drug_a
    return smiles_a, smiles_b, drug_a, drug_b


def featurize_pair(smiles_a, smiles_b, drug_a="", drug_b=""):
    """Compute the full feature vector for one drug pair.

    The pair is canonically ordered first, so the vector -- and therefore the
    prediction -- does not depend on which drug the caller listed first.

    Raises UnresolvedStructure if RDKit is missing or either SMILES fails to
    parse. An all-zero structural block is indistinguishable, to the model, from
    a real molecule with no bits set -- so it must never be fabricated.
    """
    smiles_a, smiles_b, drug_a, drug_b = _canonical_order(
        smiles_a, smiles_b, drug_a, drug_b)

    if not RDKIT_AVAILABLE:
        raise UnresolvedStructure(
            "RDKit is required to featurize drug pairs but is not installed. "
            "Install it with: pip install rdkit")
    if not smiles_a or not smiles_b:
        missing = drug_a if not smiles_a else drug_b
        raise UnresolvedStructure(f"No SMILES resolved for {missing!r}")

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        bad = drug_a if mol_a is None else drug_b
        raise UnresolvedStructure(f"RDKit could not parse the SMILES for {bad!r}")

    parts = [
        _ecfp(mol_a), _ecfp(mol_b),
        _maccs(mol_a), _maccs(mol_b),
        _physchem(mol_a), _physchem(mol_b),
        _cyp_block(drug_a, drug_b),
    ]
    vec = np.concatenate(parts)
    tanimoto = _tanimoto(mol_a, mol_b)
    overlap = float(target_overlap(drug_a, drug_b))
    vec = np.concatenate([vec, [tanimoto, overlap]])
    assert vec.shape[0] == N_FEATURES, (vec.shape[0], N_FEATURES)
    return vec


def featurize_batch(pairs):
    """pairs: list of (smiles_a, smiles_b, name_a, name_b)."""
    return np.vstack([featurize_pair(*p) for p in pairs])
