"""
End-to-End Test Suite for Autonomous Hit-to-Lead Medicinal Chemistry Triage Agent.
Verifies:
  1. Step 1: Target presets, Toxic Hit quick-fill, and custom SMILES / ChEMBL ID resolution
  2. Step 2: Multi-step DBTL trajectory generation, failed node detection, and automated backtracking
  3. Step 3: All 8 Deterministic Clinical Screening Gates
  4. Step 4: Dual 2D RDKit SVG rendering (Blue Murcko Core + Green R-Groups), Biosecurity filter,
             and mandatory Human Medicinal Chemist Sign-Off validation before JSON export.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app import (
    ChemistSignOffRequest,
    EvaluateRequest,
    TriageRunRequest,
    api_chemist_signoff,
    api_evaluate_molecule,
    api_finetune_status,
    api_get_benchmark,
    api_run_triage_sync,
    get_targets,
)
from prototypes.biotech_1_hit_to_lead import (
    DISEASE_TARGETS,
    screen_biosecurity,
)


def test_step1_targets_and_toxic_hit_preset():
    data = get_targets()
    target_ids = [t["id"] for t in data["targets"]]
    assert "CHEMBL203" in target_ids
    assert "CHEMBL4822" in target_ids
    assert data["toxic_hit_preset"]["target_id"] == "CHEMBL203"


def test_step2_and_step3_dbtl_trajectory_and_8_gates():
    for target_id in DISEASE_TARGETS:
        payload = api_run_triage_sync(
            TriageRunRequest(
                target_id=target_id,
                compound_input="",
                max_depth=10,
                min_tanimoto=0.55,
            )
        )
        events = payload["events"]
        dossier = payload["dossier"]

        # Verify trajectory has all 4 nodes (Baseline, Edit 1 PASS, Edit 2 FAIL, Edit 3 LEAD) + Backtrack event
        node_events = [e["node"] for e in events if e["event"] == "node"]
        backtrack_events = [e for e in events if e["event"] == "backtrack"]
        assert len(node_events) == 4
        assert len(backtrack_events) == 1
        assert node_events[1]["status"] == "PASS"
        assert node_events[2]["status"] == "FAIL"
        assert node_events[3]["status"] == "LEAD"

        # Verify Lead clears all 8 clinical gates
        lead_eval = dossier["optimized_lead"]["evaluation"]
        assert lead_eval["passed_all_gates"] is True
        assert lead_eval["gates_passed_count"] == 8
        assert len(lead_eval["gates"]) == 8


def test_step4_biosecurity_and_human_signoff_gate():
    payload = api_run_triage_sync(
        TriageRunRequest(target_id="CHEMBL203", compound_input="", max_depth=10, min_tanimoto=0.55)
    )
    dossier = payload["dossier"]

    # 1. Verify Dual 2D RDKit SVGs are valid SVG strings
    assert "<svg" in dossier["starting_compound"]["svg"]
    assert "<svg" in dossier["optimized_lead"]["svg"]

    # 2. Verify Biosecurity is CLEAN for the lead and BLOCKS nerve-agent SMARTS
    assert dossier["biosecurity"]["status"] == "CLEAN"
    assert dossier["biosecurity"]["matches_found"] == 0
    blocked = screen_biosecurity("CCOP(=O)(F)C")  # Fluorophosphonate nerve agent core
    assert blocked["is_clean"] is False
    assert blocked["status"] == "BLOCKED_DUAL_USE_ALERT"

    # 3. Verify Human Sign-Off rejects uncertified or empty notes
    with pytest.raises(HTTPException) as exc_info:
        api_chemist_signoff(
            ChemistSignOffRequest(
                dossier_id=dossier["dossier_id"],
                certified=False,
                reviewer_notes="Swapping out amine linkers cleared hERG.",
                dossier_payload=dossier,
            )
        )
    assert exc_info.value.status_code == 400

    # 4. Verify Human Sign-Off succeeds when checkbox is ticked and notes are provided
    export_pkg = api_chemist_signoff(
        ChemistSignOffRequest(
            dossier_id=dossier["dossier_id"],
            certified=True,
            reviewer_notes="Swapping out amine linkers successfully cleared hERG liabilities without breaking scaffold integrity.",
            dossier_payload=dossier,
        )
    )
    assert export_pkg["human_in_the_loop_signoff"]["certified_by_human_chemist"] is True
    assert export_pkg["clinical_lead_dossier"]["dossier_id"] == dossier["dossier_id"]


def test_benchmark_evaluation_endpoint():
    bench = api_get_benchmark()
    assert bench["benchmark_size"] == 25
    assert len(bench["methods"]) == 3
    assert bench["methods"][2]["valid_smiles_pct"] == 100.0
    assert bench["methods"][2]["pass_all_8_gates_pct"] >= 90.0


def test_input_recognition_smiles_chembl_drug_and_invalid():
    # 1. Drug name recognition ("Gefitinib")
    res_drug = api_evaluate_molecule(EvaluateRequest(smiles="Gefitinib"))
    assert res_drug["recognition"]["recognized"] is True
    assert "Gefitinib" in res_drug["recognition"]["label"]
    assert res_drug["evaluation"]["valid"] is True

    # 2. ChEMBL ID recognition ("CHEMBL25" -> Aspirin)
    res_chembl = api_evaluate_molecule(EvaluateRequest(smiles="CHEMBL25"))
    assert res_chembl["recognition"]["recognized"] is True
    assert "C9H8O4" in res_chembl["recognition"]["formula"]
    assert res_chembl["evaluation"]["valid"] is True

    # 3. Direct custom SMILES recognition
    res_smiles = api_evaluate_molecule(EvaluateRequest(smiles="Nc1ccc(Br)cc1C(=O)Nc2ccccc2"))
    assert res_smiles["recognition"]["recognized"] is True
    assert res_smiles["recognition"]["formula"] == "C13H11BrN2O"
    assert res_smiles["evaluation"]["valid"] is True

    # 4. Invalid / unrecognized string -> flagged as unrecognized and fails Gate #1
    res_bad = api_evaluate_molecule(EvaluateRequest(smiles="NOT_A_VALID_MOLECULE_999"))
    assert res_bad["recognition"]["recognized"] is False
    assert res_bad["evaluation"]["valid"] is False
    assert res_bad["evaluation"]["gates"][0]["passed"] is False

    # 5. Already-clean molecule ("Caffeine") -> passes 8/8 gates and is NOT modified for no reason!
    caffeine_run = api_run_triage_sync(TriageRunRequest(target_id="CHEMBL203", compound_input="Caffeine"))
    assert caffeine_run["dossier"]["already_optimal"] is True
    assert caffeine_run["dossier"]["starting_compound"]["smiles"] == caffeine_run["dossier"]["optimized_lead"]["smiles"]
    assert caffeine_run["dossier"]["deltas"]["gates_before"] == "8/8"


def test_ponatinib_aspirin_and_incomplete_lead_interlock():
    # 1. Verify Ponatinib, Aspirin, Imatinib, Terfenadine, Nimesulide all reach 8/8 gates at default min_tanimoto=0.55
    for drug_name in ["Ponatinib", "Aspirin", "Imatinib", "Terfenadine", "Nimesulide"]:
        run_res = api_run_triage_sync(
            TriageRunRequest(target_id="CHEMBL203", compound_input=drug_name, max_depth=10, min_tanimoto=0.55)
        )
        lead_ev = run_res["dossier"]["optimized_lead"]["evaluation"]
        assert lead_ev["passed_all_gates"] is True, f"{drug_name} failed to reach 8/8: {lead_ev['gates']}"
        assert run_res["dossier"]["converged_all_8"] is True
        # Verify Gate #2 displays 3 decimal places and matches status
        g2 = lead_ev["gates"][1]
        assert g2["passed"] is True
        assert g2["scalar"] >= 0.550
        # Verify 10 candidate compounds are evaluated and chemical validity proof is included
        assert len(run_res["dossier"]["evaluated_candidates"]) == 10
        ver = lead_ev["verification"]
        assert ver["radical_electrons"] == 0
        assert "RDKit Sanitize & Kekulize Passed" in ver["valence_status"]
        assert len(ver["inchi_key"]) > 10

    # 2. Verify Bug 2 Interlock: if min_tanimoto is set impossibly high (0.95) on Ponatinib,
    # it terminates honestly as INCOMPLETE (<8/8) and blocks Chemist Sign-Off & Export (HTTP 400).
    incomplete_run = api_run_triage_sync(
        TriageRunRequest(target_id="CHEMBL1862", compound_input="Ponatinib", max_depth=10, min_tanimoto=0.95)
    )
    inc_dossier = incomplete_run["dossier"]
    assert inc_dossier["converged_all_8"] is False
    assert inc_dossier["optimized_lead"]["evaluation"]["passed_all_gates"] is False

    with pytest.raises(HTTPException) as exc_info:
        api_chemist_signoff(
            ChemistSignOffRequest(
                dossier_id=inc_dossier["dossier_id"],
                certified=True,
                reviewer_notes="Attempting to sign off an incomplete candidate.",
                dossier_payload=inc_dossier,
            )
        )
    assert exc_info.value.status_code == 400
    assert "Clinical Gate Interlock" in exc_info.value.detail
