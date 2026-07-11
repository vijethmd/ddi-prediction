"""Memory-safe streaming parser for the DrugBank full-database XML.

The full file is ~1.9 GB; loading it into a DOM would exhaust memory, so we use
iterparse and clear each drug element after extracting its interactions.

Two extraction paths are provided:
  parse_drugbank()        -> DDI pairs CSV (drug, partner, description)
  extract_drug_reference()-> per-drug SMILES + CYP450 profile JSON

The reference extraction lets the whole training pipeline run offline: SMILES
strings come straight from the XML rather than the PubChem network API.
"""

import csv
import json
import re
import xml.etree.ElementTree as ET

NS = "{http://www.drugbank.ca}"

_CYP_TOKENS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]

# DrugBank spells the enzyme <name> out in full ("Cytochrome P450 3A4") and only
# carries the CYP<n><L><n> symbol in <polypeptide><gene-name>. Match either form.
_CYP_LONGFORM = re.compile(r"CYTOCHROME\s*P\s*-?\s*450\s*([0-9]+[A-Z]+[0-9]+)")


def _find_smiles(drug_elem):
    """Return the SMILES string from a drug element, or '' if absent.

    DrugBank stores it under calculated-properties (and sometimes
    experimental-properties) as a kind/value pair.
    """
    for block in ("calculated-properties", "experimental-properties"):
        props = drug_elem.find(f"{NS}{block}")
        if props is None:
            continue
        for prop in props.findall(f"{NS}property"):
            kind = prop.find(f"{NS}kind")
            value = prop.find(f"{NS}value")
            if kind is not None and value is not None and kind.text == "SMILES":
                return value.text or ""
    return ""


def _normalise_isoform(*texts):
    """Map any DrugBank spelling of a CYP enzyme onto its canonical symbol.

    Accepts both the gene symbol ('CYP3A4') and the long form DrugBank uses in
    <name> ('Cytochrome P450 3A4'). Returns None for non-CYP enzymes.
    """
    for text in texts:
        if not text:
            continue
        upper = text.upper().replace("-", " ")
        match = _CYP_LONGFORM.search(upper)
        if match:
            candidate = "CYP" + match.group(1)
            if candidate in _CYP_TOKENS:
                return candidate
        squashed = upper.replace(" ", "")
        for token in _CYP_TOKENS:
            if token in squashed:
                return token
    return None


def _find_synonyms(drug_elem):
    """Return the drug's synonyms, lowercased.

    DrugBank keys every drug on its primary <name>, which is often the systematic
    one ('Acetylsalicylic acid'), while users and prescribers type the common one
    ('Aspirin'). Indexing synonyms is what lets the offline reference table serve
    real queries without falling back to the PubChem network.
    """
    out = set()
    syns = drug_elem.find(f"{NS}synonyms")
    if syns is None:
        return []
    for syn in syns.findall(f"{NS}synonym"):
        if syn.text:
            text = syn.text.strip().lower()
            if text:
                out.add(text)
    return sorted(out)


def _find_targets(drug_elem):
    """Return the drug's protein targets as UniProt accessions.

    DrugBank nests the accession on <polypeptide id="...">; the <target><id> is
    a DrugBank-internal BE identifier, so the polypeptide id is the one that is
    comparable across drugs. Falls back to the gene symbol when absent.
    """
    accessions = set()
    targets = drug_elem.find(f"{NS}targets")
    if targets is None:
        return []
    for target in targets.findall(f"{NS}target"):
        poly = target.find(f"{NS}polypeptide")
        if poly is None:
            continue
        acc = poly.get("id")
        if not acc:
            gene = poly.find(f"{NS}gene-name")
            acc = gene.text if gene is not None else None
        if acc:
            accessions.add(acc.strip())
    return sorted(accessions)


def _find_cyp_profile(drug_elem):
    """Return the drug's CYP450 role profile across the five major isoforms.

    DrugBank lists metabolising and inhibited enzymes under <enzymes>. The gene
    symbol lives at enzyme/polypeptide/gene-name, NOT directly on <enzyme>; the
    enzyme's own <name> carries only the long form. Each enzyme has <actions>:
    'substrate' means the drug is cleared by that isoform, 'inhibitor' means it
    blocks it, 'inducer' means it upregulates it. An enzyme listed with no
    explicit action implies substrate.

    The <inhibition-strength> / <induction-strength> fields are graded (strong /
    moderate / weak) and are retained so downstream features can weight a strong
    inhibitor above a weak one.
    """
    inhibits, substrate, induces = set(), set(), set()
    strength = {}
    enzymes = drug_elem.find(f"{NS}enzymes")
    if enzymes is None:
        return {"inhibits": [], "substrate": [], "induces": [], "strength": {}}

    for enzyme in enzymes.findall(f"{NS}enzyme"):
        name_el = enzyme.find(f"{NS}name")
        poly = enzyme.find(f"{NS}polypeptide")
        gene_el = poly.find(f"{NS}gene-name") if poly is not None else None
        iso = _normalise_isoform(
            gene_el.text if gene_el is not None else None,
            name_el.text if name_el is not None else None,
        )
        if iso is None:
            continue

        actions = enzyme.find(f"{NS}actions")
        action_texts = []
        if actions is not None:
            action_texts = [a.text.lower() for a in actions.findall(f"{NS}action")
                            if a.text]
        if any("inhibitor" in a for a in action_texts):
            inhibits.add(iso)
        if any("substrate" in a for a in action_texts):
            substrate.add(iso)
        if any("inducer" in a for a in action_texts):
            induces.add(iso)
        if not action_texts:
            substrate.add(iso)

        for tag, role in (("inhibition-strength", "inhibits"),
                          ("induction-strength", "induces")):
            el = enzyme.find(f"{NS}{tag}")
            if el is not None and el.text:
                strength[f"{iso}:{role}"] = el.text.strip().lower()

    return {
        "inhibits": sorted(inhibits),
        "substrate": sorted(substrate),
        "induces": sorted(induces),
        "strength": strength,
    }


def parse_drugbank(xml_path, out_csv, progress_every=1000, logger=print):  # pragma: no cover
    """Stream the DrugBank XML and write one row per drug-interaction pair.

    Columns: drug_id, drug_name, partner_id, partner_name, description.
    Returns the number of interaction rows written.
    """
    rows_written = 0
    drugs_seen = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["drug_id", "drug_name", "partner_id", "partner_name", "description"]
        )

        context = ET.iterparse(xml_path, events=("end",))
        for event, elem in context:
            if elem.tag != f"{NS}drug":
                continue

            db_id_el = elem.find(f"{NS}drugbank-id[@primary='true']")
            name_el = elem.find(f"{NS}name")
            if db_id_el is None or name_el is None:
                elem.clear()
                continue

            drug_id = db_id_el.text or ""
            drug_name = name_el.text or ""

            interactions = elem.find(f"{NS}drug-interactions")
            if interactions is not None:
                for ddi in interactions.findall(f"{NS}drug-interaction"):
                    pid = ddi.find(f"{NS}drugbank-id")
                    pname = ddi.find(f"{NS}name")
                    desc = ddi.find(f"{NS}description")
                    writer.writerow([
                        drug_id,
                        drug_name,
                        pid.text if pid is not None else "",
                        pname.text if pname is not None else "",
                        desc.text if desc is not None else "",
                    ])
                    rows_written += 1

            drugs_seen += 1
            if drugs_seen % progress_every == 0:
                logger(f"  processed {drugs_seen} drugs, "
                       f"{rows_written} interactions")
            elem.clear()

    logger(f"Done. {drugs_seen} drugs, {rows_written} interactions -> {out_csv}")
    return rows_written


def extract_drug_reference(xml_path, out_json, progress_every=1000,
                           logger=print):  # pragma: no cover
    """Stream the XML and build a per-drug reference table.

    Output JSON maps lowercase drug name -> {
        "drugbank_id": str,
        "smiles": str,
        "cyp": {"inhibits": [...], "substrate": [...], "induces": [...],
                "strength": {...}},
        "targets": [uniprot_accession, ...],
    }

    This is the offline replacement for PubChem name->SMILES resolution and the
    source of truth for CYP450 features across the whole DrugBank catalogue.
    Returns the number of drugs written.
    """
    reference = {}
    drugs_seen = 0
    with_smiles = 0

    context = ET.iterparse(xml_path, events=("end",))
    for event, elem in context:
        if elem.tag != f"{NS}drug":
            continue

        db_id_el = elem.find(f"{NS}drugbank-id[@primary='true']")
        name_el = elem.find(f"{NS}name")
        if db_id_el is None or name_el is None:
            elem.clear()
            continue

        name = (name_el.text or "").strip()
        if name:
            smiles = _find_smiles(elem)
            reference[name.lower()] = {
                "drugbank_id": db_id_el.text or "",
                "smiles": smiles,
                "cyp": _find_cyp_profile(elem),
                "targets": _find_targets(elem),
                "aliases": _find_synonyms(elem),
            }
            if smiles:
                with_smiles += 1

        drugs_seen += 1
        if drugs_seen % progress_every == 0:
            logger(f"  processed {drugs_seen} drugs, {with_smiles} with SMILES")
        elem.clear()

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(reference, fh)

    logger(f"Done. {drugs_seen} drugs ({with_smiles} with SMILES) -> {out_json}")
    return len(reference)
