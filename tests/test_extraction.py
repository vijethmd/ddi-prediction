import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from ddi.data.parse_drugbank import (
    parse_drugbank, extract_drug_reference, _find_smiles, _find_cyp_profile,
)
from ddi.features.pubchem import (
    load_reference_table, reference_table, resolve_smiles,
)
from ddi.data.knowledge_base import load_reference_cyp, cyp_conflicts
import xml.etree.ElementTree as ET

# Mirrors the real DrugBank layout: the CYP gene symbol lives at
# enzyme/polypeptide/gene-name, never directly on <enzyme>, and <name> carries
# only the long form. Warfarin below deliberately omits the gene symbol so the
# long-form fallback is exercised. The old fixture put <gene-name> directly under
# <enzyme>, which is why the extraction bug passed its own tests for so long.
MINI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca">
  <drug type="small molecule">
    <drugbank-id primary="true">DB00682</drugbank-id>
    <name>Warfarin</name>
    <synonyms><synonym>Coumadin</synonym></synonyms>
    <calculated-properties>
      <property><kind>SMILES</kind><value>CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O</value></property>
    </calculated-properties>
    <targets>
      <target><id>BE0000011</id><polypeptide id="P00742"><gene-name>F10</gene-name></polypeptide></target>
      <target><id>BE0000012</id><polypeptide id="Q9BQB6"><gene-name>VKORC1</gene-name></polypeptide></target>
    </targets>
    <enzymes>
      <enzyme><name>Cytochrome P450 2C9</name>
        <actions><action>substrate</action></actions></enzyme>
    </enzymes>
    <drug-interactions>
      <drug-interaction><drugbank-id>DB00196</drugbank-id><name>Fluconazole</name>
        <description>This combination is contraindicated.</description></drug-interaction>
    </drug-interactions>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB00196</drugbank-id>
    <name>Fluconazole</name>
    <calculated-properties>
      <property><kind>SMILES</kind><value>OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F</value></property>
    </calculated-properties>
    <targets>
      <target><id>BE0000013</id><polypeptide id="P00742"><gene-name>F10</gene-name></polypeptide></target>
    </targets>
    <enzymes>
      <enzyme><name>Cytochrome P450 2C9</name>
        <polypeptide id="P11712"><gene-name>CYP2C9</gene-name></polypeptide>
        <actions><action>inhibitor</action></actions>
        <inhibition-strength>strong</inhibition-strength></enzyme>
    </enzymes>
    <drug-interactions></drug-interactions>
  </drug>
</drugbank>
"""


@pytest.fixture
def mini_xml(tmp_path):
    p = tmp_path / "mini.xml"
    p.write_text(MINI_XML)
    return str(p)


def test_extract_reference_smiles_and_cyp(mini_xml, tmp_path):
    out = tmp_path / "ref.json"
    n = extract_drug_reference(mini_xml, str(out), logger=lambda m: None)
    assert n == 2
    ref = json.loads(out.read_text())
    assert ref["warfarin"]["smiles"].startswith("CC(=O)")
    assert ref["warfarin"]["cyp"]["substrate"] == ["CYP2C9"]
    assert ref["fluconazole"]["cyp"]["inhibits"] == ["CYP2C9"]


def test_parse_pairs(mini_xml, tmp_path):
    out = tmp_path / "pairs.csv"
    n = parse_drugbank(mini_xml, str(out), logger=lambda m: None)
    assert n == 1
    text = out.read_text()
    assert "Warfarin" in text and "Fluconazole" in text


def test_offline_chain_from_reference(mini_xml, tmp_path):
    ref_path = tmp_path / "ref.json"
    extract_drug_reference(mini_xml, str(ref_path), logger=lambda m: None)

    n = load_reference_table(str(ref_path))
    assert n == 2
    ncyp = load_reference_cyp(reference_table())
    assert ncyp == 2

    # SMILES resolves offline from the reference table.
    smi = resolve_smiles("warfarin", allow_network=False)
    assert smi and smi.startswith("CC(=O)")

    # CYP conflict is derived from the extracted profiles.
    conflicts = cyp_conflicts("warfarin", "fluconazole")
    assert ("CYP2C9", "fluconazole", "warfarin") in conflicts

    # Reset global state so other tests are unaffected.
    load_reference_table.__wrapped__ if hasattr(load_reference_table, "__wrapped__") else None


def test_find_smiles_missing():
    root = ET.fromstring(
        '<drug xmlns="http://www.drugbank.ca"><name>X</name></drug>')
    assert _find_smiles(root) == ""


def test_find_cyp_no_enzymes():
    root = ET.fromstring(
        '<drug xmlns="http://www.drugbank.ca"><name>X</name></drug>')
    prof = _find_cyp_profile(root)
    assert prof == {"inhibits": [], "substrate": [], "induces": [],
                    "strength": {}}


def test_cyp_isoform_read_from_nested_polypeptide():
    """Regression: the gene symbol is at enzyme/polypeptide/gene-name."""
    root = ET.fromstring(
        '<drug xmlns="http://www.drugbank.ca"><enzymes><enzyme>'
        '<name>Some Enzyme</name>'
        '<polypeptide id="P08684"><gene-name>CYP3A4</gene-name></polypeptide>'
        '<actions><action>inhibitor</action></actions>'
        '</enzyme></enzymes></drug>')
    assert _find_cyp_profile(root)["inhibits"] == ["CYP3A4"]


def test_cyp_isoform_read_from_longform_name():
    """Regression: DrugBank writes 'Cytochrome P450 3A4', not 'CYP3A4'."""
    root = ET.fromstring(
        '<drug xmlns="http://www.drugbank.ca"><enzymes><enzyme>'
        '<name>Cytochrome P450 3A4</name>'
        '<actions><action>substrate</action></actions>'
        '</enzyme></enzymes></drug>')
    assert _find_cyp_profile(root)["substrate"] == ["CYP3A4"]


def test_non_cyp_enzyme_ignored():
    root = ET.fromstring(
        '<drug xmlns="http://www.drugbank.ca"><enzymes><enzyme>'
        '<name>Myeloperoxidase</name>'
        '<polypeptide id="P05164"><gene-name>MPO</gene-name></polypeptide>'
        '<actions><action>inhibitor</action></actions>'
        '</enzyme></enzymes></drug>')
    prof = _find_cyp_profile(root)
    assert prof["inhibits"] == [] and prof["substrate"] == []


def test_inhibition_strength_captured(mini_xml, tmp_path):
    out = tmp_path / "ref.json"
    extract_drug_reference(mini_xml, str(out), logger=lambda m: None)
    ref = json.loads(out.read_text())
    assert ref["fluconazole"]["cyp"]["strength"]["CYP2C9:inhibits"] == "strong"


def test_targets_and_overlap(mini_xml, tmp_path):
    from ddi.data.knowledge_base import target_overlap
    out = tmp_path / "ref.json"
    extract_drug_reference(mini_xml, str(out), logger=lambda m: None)
    ref = json.loads(out.read_text())
    assert ref["warfarin"]["targets"] == ["P00742", "Q9BQB6"]
    load_reference_table(str(out))
    load_reference_cyp(reference_table())
    # Both drugs list UniProt P00742; nothing else is shared.
    assert target_overlap("warfarin", "fluconazole") == 1
    assert target_overlap("warfarin", "nonexistent") == 0


def test_synonym_resolves_to_primary_entry(mini_xml, tmp_path):
    out = tmp_path / "ref.json"
    extract_drug_reference(mini_xml, str(out), logger=lambda m: None)
    n = load_reference_table(str(out))
    assert n == 2  # counts primary drugs, not alias keys
    # 'Coumadin' is a synonym of Warfarin and must resolve to its SMILES.
    assert resolve_smiles("coumadin", allow_network=False) == \
        resolve_smiles("warfarin", allow_network=False)


@pytest.fixture(autouse=True)
def _reset_reference_state():
    """Ensure reference-table globals don't leak between tests."""
    yield
    load_reference_table("/nonexistent/path.json")
    resolve_smiles.cache_clear()
    from ddi.data import knowledge_base as kb
    kb._REFERENCE_CYP = {}
    kb._REFERENCE_TARGETS = {}
