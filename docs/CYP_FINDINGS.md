# CYP450 features: what the corrected extraction actually shows

This note records what changed when the CYP450 extraction bug was fixed, and
what the corrected features do and do not support. It is the evidence base for
the claims in the paper and the README.

## The bug

`_find_cyp_profile` read the enzyme gene symbol from `enzyme/gene-name`, which
does not exist in the DrugBank schema. The symbol lives at
`enzyme/polypeptide/gene-name`, and the enzyme's own `<name>` carries only the
long form ("Cytochrome P450 3A4", not "CYP3A4"). Both lookups missed, so every
one of the 19,842 drugs received an empty CYP profile. The 25-column CYP block
in the 50,000-row training matrix was 0.16% non-zero — all of it from 22
hand-written entries in `knowledge_base.CYP_PROFILES`, none from DrugBank.

After the fix: **1,413 drugs** carry CYP data (CYP3A4 1272, CYP2D6 533,
CYP2C9 458, CYP1A2 387, CYP2C19 378). On the 50k sample, 26.6% of pairs have
CYP data on both drugs and 17.7% carry at least one inhibitor/substrate
conflict. The CYP block is now 11.5% non-zero.

Coverage is 1,413 of 19,842 drugs (7%), NOT "the full catalogue". DrugBank
simply has no enzyme record for most entries. The paper must not claim
full-catalogue CYP coverage.

## Finding 1 — CYP conflicts predict the Moderate class, not Severe

Per-class SHAP on the retrained Stage-2 classifier (mean |SHAP| over 3,000
test rows, pruned 1,339-feature space):

| Stage-2 class | CYP block share | Best CYP feature rank | Top feature |
|---|---|---|---|
| **Moderate** | **17.4%** | **1 / 1339** | `cyp_CYP3A4_b_substrate` |
| Severe | 1.7% | 38 / 1339 | `maccs_a_54` (structural) |
| Synergistic | 3.9% | 3 / 1339 | `maccs_b_110` |
| Antagonistic | 2.4% | 8 / 1339 | `maccs_b_142` |

The top three features for Moderate are all CYP3A4 (substrate a, substrate b,
conflict). For Severe the model leans on MACCS structural keys instead.

This is not a modelling artefact — it is baked into the labels. On the 50k
sample:

| | P(Severe) | P(Moderate) |
|---|---|---|
| base rate | 24.1% | 65.1% |
| given a CYP conflict | **9.7%** | **88.2%** |

A CYP conflict makes a pair 2.5x *less* likely to be Severe. The reason is the
weak-supervision labeller: a CYP conflict produces DrugBank text like "The
metabolism of X can be decreased when combined with Y", which the labeller maps
to Moderate on the word "metabolism". Severe is reserved for named adverse
effects (bleeding, QT prolongation, CNS depression), which are pharmacodynamic
and carry no CYP conflict. 79% of conflict pairs have pharmacokinetic
description text vs 27.6% of non-conflict pairs.

**Consequence for the paper:** the claim that CYP features reproduce a "34%
Severe-importance figure" is false under these labels and cannot be rescued by
any code change short of redefining the labels. The defensible, and arguably
stronger, claim is that CYP conflict features are the dominant driver of the
*pharmacokinetic* (Moderate) severity class.

### On circularity

Because the CYP feature and the severity label are both derived from the same
DrugBank record, CYP's high Moderate-importance is partly the model recovering
DrugBank's own annotation from a second field. This is disclosed as a threat to
validity. Finding 2 is the answer to it.

## Finding 2 — CYP features earn their place on cold-start

Drug-disjoint ablation (`scripts/coldstart_ablation.py`): hold out 25% of drugs
entirely; test only on pairs where BOTH drugs are unseen. Compare feature
groups by zeroing them. Numbers are from reduced-size forests for runtime, so
they are lower than the deployed bundle — the comparison across variants is
what matters, not the absolute level.

| variant | random macro-F1 | cold-start macro-F1 | cold-start Severe PR-AUC |
|---|---|---|---|
| full | 0.373 | 0.319 | 0.540 |
| no_cyp | 0.377 | 0.295 | 0.487 |
| no_fingerprint | 0.284 | 0.242 | 0.442 |
| cyp_only (27 feats) | 0.151 | 0.170 | 0.309 |

Reading:

- **On the random split, CYP contributes nothing** (full 0.373 vs no_cyp 0.377).
  A model can memorise "anything with warfarin is Severe" from fingerprint bits
  when the drug is in the training set.
- **On cold-start, removing CYP costs 0.024 macro-F1 and 0.053 Severe PR-AUC.**
  When the drug is unseen, its fingerprint is uninformative but its enzyme
  profile still carries signal.
- **`cyp_only` is the only variant that improves from random to cold-start**
  (0.151 -> 0.170): the 27 pharmacology features do not overfit to drug identity
  the way 2,382 fingerprint bits do.

This is a modest but real effect, and it is the honest version of the paper's
cold-start motivation. It also answers the circularity concern: for a genuinely
novel drug you have in-vitro enzyme data but no DrugBank interaction text to
leak from, so CYP's cold-start value is not the same recovered annotation.

## Finding 3 — retrained model, corrected data (deployed bundle)

50k deduplicated pairs, canonical drug ordering, real CYP + target features.
Held-out test fold (n=10,000):

| Metric | Old bundle | New bundle |
|---|---|---|
| Severe recall, argmax | 0.512 | 0.674 |
| Severe recall, thresholded (tau=0.134) | 0.941 | 0.922 |
| Macro F1, thresholded | 0.353 | 0.493 |
| Weighted F1, thresholded | 0.533 | 0.675 |
| Cohen kappa, thresholded | — | 0.421 |
| Severe PR-AUC | 0.692 | 0.800 |
| Moderate PR-AUC | 0.870 | 0.916 |

Per-class calibration (ECE raw -> calibrated): No Interaction 0.038->0.005,
Moderate 0.133->0.014, Severe 0.095->0.012, Synergistic 0.002->0.002,
Antagonistic 0.013->0.012.

## Finding 4 — the conformal sets are wide

Mean prediction-set size 3.56 of 5; **singleton rate 0.3%**; 18.8% of
predictions return all five classes. Class-conditional coverage holds
(Severe 0.953 vs 0.95 target, Moderate 0.906 vs 0.90), but it is achieved by
returning large sets, not by confident singletons.

The paper's statement that "most sets are singletons" is false and must be
removed. Correct framing: coverage is guaranteed and holds empirically; the
sets are informative only for the small fraction of pairs where they are small,
and a full five-class set is an honest "don't know".

## Data-integrity fixes bundled in

- **Mirrored duplicates.** DrugBank lists each interaction twice (once per
  drug). The raw pairs file had 2,911,156 rows for 1,455,878 unordered pairs;
  the two directions always share a label. Sampling both leaked a pair's mirror
  from train into test. Now deduplicated before sampling.
- **Order dependence.** The feature layout put drug A and drug B in fixed
  blocks, so f(A,B) != f(B,A); near tau this flipped the served class on
  argument order alone. `featurize_pair` now canonically orders the pair.
- **Silent zeros.** An unresolved SMILES used to yield an all-zero structural
  block, which the model maps to a confident Severe. Featurization now raises
  `UnresolvedStructure`; training drops the row, serving declines the model and
  says so.
