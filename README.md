# Autonomous Hit-to-Lead Medicinal Chemistry Triage Agent

**Event:** AgentVerse 2026 — Biotech Track (*"Make the path to patients faster"*)

An end-to-end **Autonomous Design-Build-Test-Learn (DBTL) Medicinal Chemistry Triage Agent** powered by **Gemini Argon**, **RDKit deterministic cheminformatics (8 clinical screening gates)**, **Bemis-Murcko `[cD2]` scaffold locking**, **10-candidate multi-step beam search with automated backtracking**, **5-point chemical validity verification**, and a **Dual-Use Biosecurity & Mandatory Human Medicinal Chemist Sign-Off Interlock**.

---

## Included Pitch Deck & Technical Documentation

- **PowerPoint Pitch Deck (8 Widescreen Slides):** [`AgentVerse26_Hit_To_Lead_Pitch_Deck.pptx`](./AgentVerse26_Hit_To_Lead_Pitch_Deck.pptx)
- **PDF Technical Explanation (4 Pages):** [`AgentVerse26_Hit_To_Lead_Technical_Guide.pdf`](./AgentVerse26_Hit_To_Lead_Technical_Guide.pdf)
- **Architecture & Rubric Specification:** [`BIOTECH_HIT_TO_LEAD_HANDOFF.md`](./BIOTECH_HIT_TO_LEAD_HANDOFF.md)

---

## Quick Start

```powershell
# 1. Create / activate virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch the FastAPI + Interactive 3-Column Web Application
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## 4-Stage End-to-End Architecture

1. **Stage 1 — Universal Compound Recognition & Baseline Diagnosis (`1. Input & Configuration`)**
   - Accepts any **Drug Name** (`Ponatinib`, `Imatinib`, `Terfenadine`, `Aspirin`, `Caffeine`, `Nimesulide`, `Gefitinib`), **ChEMBL ID** (`CHEMBL25`), or **raw SMILES string**.
   - Resolves structures via curated lookup + live **ChEMBL v33 REST** and **PubChem PUG-REST** APIs.
   - Immediately evaluates all 8 clinical gates: if an input compound already passes `8/8` checks (e.g., `Caffeine`), the agent halts on Cycle 0 with **0 unnecessary modifications**.

2. **Stage 2 — Gemini Argon + RDKit `[cD2]` SMIRKS DBTL Search (`2. Optimization Path`)**
   - Locks the Bemis-Murcko binding core (`Tanimoto >= 0.550`) using degree-2 unsubstituted aromatic carbon (`[cD2]`) SMIRKS so ring substituents on trisubstituted cores (such as Ponatinib) are never accidentally deleted.
   - Generates and evaluates up to **Max Optimization Cycles (default 10) distinct candidate compounds**, including:
     - **Step 1:** Intermediate targeted fix
     - **Attempt 2:** Over-lipophilic / nitro-phenyl extension (`FAIL` → triggers **Automated Backtrack** to Step 1)
     - **Cycles 3–9:** Structural bioisostere & peripheral C-H substituent scan analogs
     - **Cycle 10 (Selected Lead):** Converged `8/8 PASS` Optimized Lead
   - Clicking any of the 10 evaluated candidate compounds in Column 2 loads its 2D SVG depiction, 8-gate scorecard, and validity proof.

3. **Stage 3 — 8 Deterministic Clinical Screening Gates & Chemical Validity Proof (`3. Clinical Safety Checks`)**
   - **Gate 1 — Valid Structure:** Clean RDKit `SanitizeMol` & Kekulization (`0 valence errors, 0 radicals`)
   - **Gate 2 — Core Shape Kept:** Blended Morgan ECFP4 (70%) + Murcko Core (30%) `Tanimoto >= 0.550` (displayed & compared to 3 decimal places, synced live to the UI slider) + intact Murcko ring count
   - **Gate 3 — Oral Drug Size (Lipinski Rule of 5):** `0 Violations` (`MW <= 500 Da`, `cLogP <= 5.0`, `HBD <= 5`, `HBA <= 10`)
   - **Gate 4 — Cell Absorption (Ertl TPSA):** `40.0 Å² <= TPSA <= 140.0 Å²`
   - **Gate 5 — Water Solubility (Delaney ESOL):** `LogS > -5.0 log mol/L`
   - **Gate 6 — Heart Safety (hERG Cardiotoxicity):** `Probability < 0.45`
   - **Gate 7 — No Toxic Groups (NIH PAINS + Brenk + Toxicophore SMARTS):** `0 Alerts`
   - **Gate 8 — Lab Synthesizability (Ertl SAScore):** `SAScore <= 4.0`
   - **5-Point New Molecule Chemical Validity Proof:** Displays `RDKit Sanitize & Kekulize Passed (0 valence errors, 0 radicals)`, Hill Molecular Formula, Exact Monoisotopic MW, IUPAC Standard InChIKey, one-click **`Verify as New Input`** (re-evaluates the generated SMILES from scratch), and direct **`Verify in PubChem`** link.

4. **Stage 4 — Biosecurity Screen, Hard `<8/8` Interlock & Mandatory Human Sign-Off**
   - Side-by-side 2D RDKit SVGs highlighting the **Preserved Core Scaffold (Blue)**, **Modified R-Groups (Green)**, and **Toxic Alerts (Red)**.
   - Deterministic SMARTS biosecurity filter screening against restricted organophosphorus nerve agents, sulfur/nitrogen mustards, and carbamate toxins (`0 matches found`).
   - **Hard Clinical Gate Interlock:** If a run cannot reach `8/8` within the user's similarity constraint, the agent reports the incomplete status honestly, disables `Approve & Export JSON`, and blocks `POST /api/signoff-export` with HTTP `400`.

---

## Verification & Automated Tests

```powershell
# Run the full 6-part end-to-end pytest suite
.\.venv\Scripts\python.exe -m pytest -v test_end_to_end.py

# Run the 25-scaffold quantitative benchmark evaluation
.\.venv\Scripts\python.exe evaluate_benchmark.py
```

---

## Data Sources & Open-Source Licenses

| Data Source / Library | License | Role in Pipeline |
| :--- | :---: | :--- |
| **ChEMBL Database (REST API v33)** | CC-BY-SA 3.0 | Target metadata (`CHEMBL203`, `CHEMBL1862`, `CHEMBL4822`, etc.) & bioactivity structures |
| **RDKit Cheminformatics (`2026.09`)** | BSD-3-Clause | Sanitization, Kekulization, InChIKey, ECFP4 fingerprints, PAINS/Brenk catalogs & SMIRKS reactions |
| **PubChem PUG-REST API (NCBI)** | Public Domain | Live drug-name resolution & independent structure verification |
