"""
FastAPI Web Server for Autonomous Hit-to-Lead Medicinal Chemistry Triage Agent.
Provides:
  - GET  /                     : Serves the interactive 3-column UI + Biosecurity Modal
  - GET  /api/targets          : Returns preset disease targets and toxic hit molecules
  - POST /api/evaluate         : Evaluates any single SMILES string across the 8 clinical gates + 2D SVG
  - GET  /api/stream-triage    : Server-Sent Events (SSE) real-time stream of the DBTL agent trajectory
  - POST /api/run-triage       : Synchronous JSON execution of the DBTL agent trajectory
  - POST /api/signoff-export   : Validates Human-in-the-Loop Medicinal Chemist sign-off & returns audited JSON package
  - GET  /api/benchmark        : Runs/returns the 25-compound ChEMBL baseline benchmark comparison
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from evaluate_benchmark import run_benchmark_suite
from medchem_finetune import (
    METRICS_PATH,
    get_finetuned_policy,
    reload_finetuned_policy,
    train_sft_and_grpo,
)
from prototypes.biotech_1_hit_to_lead import (
    DISEASE_TARGETS,
    KNOWN_COMPOUNDS_LOOKUP,
    evaluate_molecule,
    generate_dbtl_trajectory_events,
    recognize_compound_input,
    render_molecule_svg,
    resolve_chembl_target,
    resolve_compound_input,
    screen_biosecurity,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Autonomous Hit-to-Lead Medicinal Chemistry Triage Agent",
    description="Agentverse 2026 Biotech Track — Multi-Step DBTL Cheminformatics & Biosecurity Triage Workspace",
    version="2.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory audit log of human medicinal chemist sign-offs
SIGNOFF_AUDIT_LOG: List[Dict[str, Any]] = []


class EvaluateRequest(BaseModel):
    smiles: str
    target_id: str = "CHEMBL203"
    reference_smiles: Optional[str] = None
    min_tanimoto: float = 0.55
    mode: str = "lead"


class TriageRunRequest(BaseModel):
    target_id: str = "CHEMBL203"
    compound_input: str = ""
    max_depth: int = Field(default=10, ge=1, le=25)
    min_tanimoto: float = Field(default=0.55, ge=0.1, le=0.95)


class ChemistSignOffRequest(BaseModel):
    dossier_id: str
    certified: bool
    reviewer_notes: str
    reviewer_name: str = "Senior Medicinal Chemist (Hit-to-Lead Triage)"
    dossier_payload: Dict[str, Any]


@app.get("/api/targets")
def get_targets() -> Dict[str, Any]:
    """Return all preset ChEMBL disease targets and quick-fill flawed hit compounds."""
    return {
        "targets": list(DISEASE_TARGETS.values()),
        "default_target_id": "CHEMBL203",
        "toxic_hit_preset": {
            "target_id": "CHEMBL203",
            "compound_id": "CHEMBL203-HIT-884",
            "smiles": DISEASE_TARGETS["CHEMBL203"]["default_hit_smiles"],
            "description": DISEASE_TARGETS["CHEMBL203"]["default_hit_notes"],
        },
        "known_drugs": [
            {"name": "Gefitinib (CHEMBL553)", "input": "CHEMBL553", "smiles": KNOWN_COMPOUNDS_LOOKUP["CHEMBL553"][0]},
            {"name": "Erlotinib (CHEMBL939)", "input": "CHEMBL939", "smiles": KNOWN_COMPOUNDS_LOOKUP["CHEMBL939"][0]},
            {"name": "Sorafenib (CHEMBL1336)", "input": "CHEMBL1336", "smiles": KNOWN_COMPOUNDS_LOOKUP["CHEMBL1336"][0]},
            {"name": "Imatinib (CHEMBL941)", "input": "CHEMBL941", "smiles": KNOWN_COMPOUNDS_LOOKUP["CHEMBL941"][0]},
            {"name": "Celecoxib (CHEMBL118)", "input": "CHEMBL118", "smiles": KNOWN_COMPOUNDS_LOOKUP["CHEMBL118"][0]},
        ]
    }


@app.get("/api/resolve-target")
def api_resolve_target(target_id: str = Query("CHEMBL203")) -> Dict[str, Any]:
    """Resolve any ChEMBL target ID (preset or live from EBI ChEMBL API) and return its metadata + hit compound."""
    return resolve_chembl_target(target_id)


@app.post("/api/evaluate")
def api_evaluate_molecule(req: EvaluateRequest) -> Dict[str, Any]:
    """Recognize input (SMILES, ChEMBL ID, or Drug Name), evaluate across all 8 clinical gates, and render 2D RDKit SVG."""
    rec = recognize_compound_input(req.smiles, target_id=req.target_id)
    resolved_smiles = rec["smiles"] if rec["recognized"] else req.smiles
    ref_smiles = req.reference_smiles if (req.reference_smiles and not req.reference_smiles == req.smiles) else resolved_smiles

    ev = evaluate_molecule(resolved_smiles, ref_smiles, min_tanimoto=req.min_tanimoto)
    svg = render_molecule_svg(
        ev["smiles"] if ev["valid"] else resolved_smiles,
        ref_smiles,
        mode=req.mode
    )
    return {
        "recognition": rec,
        "evaluation": ev,
        "svg": svg,
    }


@app.post("/api/run-triage")
def api_run_triage_sync(req: TriageRunRequest) -> Dict[str, Any]:
    """Execute the full DBTL trajectory synchronously and return all events + final dossier."""
    events: List[Dict[str, Any]] = []
    final_dossier: Optional[Dict[str, Any]] = None
    for evt in generate_dbtl_trajectory_events(
        target_id=req.target_id,
        compound_input=req.compound_input,
        max_depth=req.max_depth,
        min_tanimoto=req.min_tanimoto,
        step_delay=0.0,
    ):
        events.append(evt)
        if evt["event"] == "complete":
            final_dossier = evt["dossier"]

    return {
        "events": events,
        "dossier": final_dossier,
    }


@app.get("/api/stream-triage")
async def api_stream_triage(
    target_id: str = Query("CHEMBL203"),
    compound_input: str = Query(""),
    max_depth: int = Query(10),
    min_tanimoto: float = Query(0.55),
    step_delay: float = Query(0.42),
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint that streams live DBTL trajectory nodes,
    Gemini 2.5 reasoning logs, automated backtracking, and 8-gate scoreboard updates.
    """
    async def event_publisher():
        for evt in generate_dbtl_trajectory_events(
            target_id=target_id,
            compound_input=compound_input,
            max_depth=max_depth,
            min_tanimoto=min_tanimoto,
            step_delay=0.0,
        ):
            payload = json.dumps(evt)
            yield f"data: {payload}\n\n"
            delay = max(0.0, min(1.5, step_delay))
            if delay > 0:
                await asyncio.sleep(delay)

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/signoff-export")
def api_chemist_signoff(req: ChemistSignOffRequest) -> Dict[str, Any]:
    """
    Human-in-the-Loop Medicinal Chemist Override Sign-Off Gate.
    Strictly enforces that `certified == True` and `reviewer_notes` is non-empty before unlocking export.
    """
    notes = (req.reviewer_notes or "").strip()
    if not req.certified:
        raise HTTPException(
            status_code=400,
            detail="Human Medicinal Chemist certification checkbox must be ticked before lead export."
        )
    if len(notes) < 5:
        raise HTTPException(
            status_code=400,
            detail="Reviewer Notes / Rational Keylog justification is required (minimum 5 characters)."
        )

    biosec = req.dossier_payload.get("biosecurity", {})
    if biosec.get("status") == "BLOCKED_DUAL_USE_ALERT":
        raise HTTPException(
            status_code=403,
            detail="Biosecurity Interlock: Export blocked due to dual-use toxin SMARTS match."
        )

    lead_eval = req.dossier_payload.get("optimized_lead", {}).get("evaluation", {})
    if not lead_eval.get("passed_all_gates", False):
        failed_names = [g.get("plain_name") or g.get("name") for g in lead_eval.get("gates", []) if not g.get("passed")]
        raise HTTPException(
            status_code=400,
            detail=f"Clinical Gate Interlock: Cannot approve or export a compound that did not pass all 8/8 gates (failing: {', '.join(failed_names)})."
        )

    signoff_record = {
        "signoff_id": f"SIGNOFF-{int(time.time())}",
        "dossier_id": req.dossier_id,
        "approved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "certified_by_human_chemist": True,
        "reviewer_role": req.reviewer_name,
        "reviewer_rational_keylog": notes,
        "biosecurity_status": "CLEAN (0 Restricted Toxin Matches)",
    }
    SIGNOFF_AUDIT_LOG.append(signoff_record)

    # Return the complete signed clinical lead handoff JSON package matching the engineering spec
    export_package = {
        "schema_version": "Agentverse-2026-Biotech-Lead-Handoff-v1.0",
        "human_in_the_loop_signoff": signoff_record,
        "clinical_lead_dossier": {
            "dossier_id": req.dossier_payload.get("dossier_id", req.dossier_id),
            "timestamp_utc": req.dossier_payload.get("timestamp_utc"),
            "target": req.dossier_payload.get("target"),
            "hyperparameters": req.dossier_payload.get("hyperparameters"),
            "starting_flawed_hit": {
                "label": req.dossier_payload.get("starting_compound", {}).get("label"),
                "smiles": req.dossier_payload.get("starting_compound", {}).get("smiles"),
                "evaluation": req.dossier_payload.get("starting_compound", {}).get("evaluation"),
            },
            "optimized_lead_candidate": {
                "label": req.dossier_payload.get("optimized_lead", {}).get("label"),
                "smiles": req.dossier_payload.get("optimized_lead", {}).get("smiles"),
                "evaluation": req.dossier_payload.get("optimized_lead", {}).get("evaluation"),
                "modifications_summary": req.dossier_payload.get("optimized_lead", {}).get("modifications_summary"),
            },
            "multiparameter_optimization_deltas": req.dossier_payload.get("deltas"),
            "biosecurity_screening_declaration": req.dossier_payload.get("biosecurity"),
            "dbtl_trajectory_audit_trail": [
                {
                    "id": n.get("id"),
                    "cycle": n.get("cycle"),
                    "title": n.get("title"),
                    "status": n.get("status"),
                    "smiles": n.get("smiles"),
                    "gates_passed": f"{n.get('evaluation', {}).get('gates_passed_count', 0)}/8",
                    "rationale": n.get("rationale"),
                }
                for n in req.dossier_payload.get("trajectory_nodes", [])
            ],
            "data_sources_and_licenses": req.dossier_payload.get("data_sources_and_licenses", []),
        },
    }
    return export_package


@app.get("/api/benchmark")
def api_get_benchmark() -> Dict[str, Any]:
    """Return the 25-compound ChEMBL benchmark comparison table across 3 methods."""
    return run_benchmark_suite()


@app.get("/api/finetune-status")
def api_finetune_status() -> Dict[str, Any]:
    """Return the Fine-Tuned Medicinal Chemistry Neural Policy (SFT + RL-GRPO) training metrics."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    summary = train_sft_and_grpo()
    reload_finetuned_policy()
    return summary


@app.post("/api/finetune-train")
def api_finetune_train() -> Dict[str, Any]:
    """Run Stage 1 SFT + Stage 2 RL-GRPO fine-tuning on the ChEMBL MMP dataset and reload weights."""
    summary = train_sft_and_grpo()
    reload_finetuned_policy()
    return summary


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
