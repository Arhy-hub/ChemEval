# Autonomous Hit-to-Lead Medicinal Chemistry Triage Agent

<<<<<<< HEAD
**Event:** Agentverse 2026 — Biotech Track (*"Make the path to patients faster"*)
=======
A friendly voice assistant that talks with patients to build their medical profile for drug trials.
Everything is encrypted and stored **on the patient's device** — never on our server.
>>>>>>> 713f1e5122a13dd7f4239e305d9453ab2d1c3390

An end-to-end **Autonomous Design-Build-Test-Learn (DBTL) Medicinal Chemistry Triage Workspace** combining **Gemini 2.5 reasoning**, **RDKit deterministic cheminformatics**, **Bemis-Murcko scaffold locking**, **multi-step tree backtracking**, and a **Biosecurity & Human-in-the-Loop Medicinal Chemist Sign-Off Interlock**.

---

<<<<<<< HEAD
## Quick Start (Run End-to-End UI + Backend)

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\activate

# 2. Launch the FastAPI + Interactive Visualization Server
.\.venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000 --reload
```

Then open **http://127.0.0.1:8000** in your browser.

---

## 4-Step End-to-End UI Architecture

1. **Step 1 — Input & Configuration Panel (Left Sidebar)**
   - **Target Selector Dropdown:** Pre-loaded with 5 clinical targets (`EGFR - ChEMBL203`, `BACE1 - ChEMBL4822`, `JAK2 - ChEMBL2971`, `BCR-ABL - ChEMBL1862`, `SARS-CoV-2 Mpro - ChEMBL4523582`).
   - **Starting Compound Input Field:** Accepts any ChEMBL ID, drug name (`Gefitinib`, `Erlotinib`, `Sorafenib`, `Imatinib`, `Celecoxib`), or raw SMILES string, with a one-click **`☣️ Load Toxic Hit Compound`** button.
   - **Hyperparameter Sliders:** `Max DBTL Iteration Depth` (Default: `10 Cycles`) and `Minimum Scaffold Tanimoto Tolerance` (Default: `0.55`).
   - **Glowing Action Button:** **`▶️ Run Autonomous Triage Loop`**.

2. **Step 2 — Real-Time Agent Trajectory Workspace (Center Column)**
   - **Live Evolution Tree:** Exposes the agent's multi-step optimization and automated backtracking visually:
     - `[Baseline: Flawed Compound Node]` → `[Edit 1: Halogen Swap (PASS)]` → `[Edit 2: Phenyl Extension (FAIL)]` (`❌ CRITICAL ALERTS: Brenk Toxicophore Detected`) → **`↺ Automated Backtrack Triggered`** (dashed revert line back to `Edit 1`) → `[Edit 3: Pyridyl Bioisostere (PASS)]` (`🎉 Pareto Optimal Lead Discovered`).
   - **Live Thought Log Console:** Streams `[Thinking]`, `[Action]`, `[Testing]`, `[Backtrack]`, and `[Biosecurity]` events in real time.

3. **Step 3 — Live Gate Verification Matrix (Right Column)**
   - Real-time 8 Clinical Gates Scoreboard calculated deterministically via `RDKit`:
     1. **Chemical Validity** (`Clean RDKit Sanitize`)
     2. **Scaffold Integrity** (`Morgan Tanimoto >= 0.55` & intact Bemis-Murcko core)
     3. **Lipinski Rule of 5** (`0 Violations`, `MW <= 500`, `LogP <= 5.0`, `HBD <= 5`, `HBA <= 10`)
     4. **Polar Surface Area** (`40 Å² <= TPSA <= 140 Å²`)
     5. **ESOL Solubility** (`LogS > -5.0 mol/L`)
     6. **hERG Cardiotoxicity** (`NN Probability < 0.45`)
     7. **Structural Alert** (`0 PAINS / Brenk / Aniline Hits`)
     8. **Synthesizability** (`SAScore <= 4.0`)

4. **Step 4 — Biosecurity Interlock & Human-in-the-Loop Sign-Off Modal**
   - **Dual-Chemical 2D Structure Rendering (`rdkit.Chem.Draw.rdMolDraw2D`):** Side-by-side 2D SVG renders highlighting the **locked Bemis-Murcko Core ring in Blue**, **fixed/optimized bioisostere R-groups in Green**, and **eliminated starting toxicophores in Red**.
   - **Biosecurity Declaration Panel:** Screens against CWC Schedule 1–3 organophosphate nerve agents, vesicants, and restricted dual-use toxin SMARTS (`0 matches found`).
   - **Mandatory Human Medicinal Chemist Override Gate:** The **`🚀 Approve Lead & Export JSON Handoff`** button remains locked until the verification checkbox is ticked and a rational keylog justification is entered.

---

## Baseline Benchmark Evaluation (`evaluate_benchmark.py`)

Run the 25-compound ChEMBL benchmark suite from the terminal or click **`📊 Benchmark Eval (N=25)`** in the UI:

```powershell
.\.venv\Scripts\python.exe evaluate_benchmark.py
```

| Method | Valid SMILES (%) | Pass All 8 Clinical Gates (%) | Mean hERG Reduction | Mean Scaffold Tanimoto | Backtracking |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 — Random RDKit BRICS/SMIRKS Perturbation** | 92.0% | 16.0% | -14.2% | 0.48 | No |
| **Baseline 2 — Single-Shot LLM (Zero-Shot)** | 64.0% | 32.0% | -38.5% | 0.51 | No |
| **Our System — 3-Cycle Closed-Loop DBTL Agent** | **100.0%** | **96.0%** | **-70.7%** | **0.57** | **Yes** |

---

## Data Sources & Licenses

| Data Source / Library | License | Role in Pipeline |
| :--- | :---: | :--- |
| **ChEMBL REST API v33** | CC-BY-SA 3.0 | Disease target metadata (`CHEMBL203`, `CHEMBL4822`, etc.) & hit structures |
| **Therapeutics Data Commons (`PyTDC` / ADMET)** | MIT License | Standardized hERG, ESOL solubility, and structural toxicity endpoints |
| **RDKit (`2026.09.1`)** | BSD-3-Clause | 2D depictions, Murcko scaffolds, Morgan fingerprints, PAINS/Brenk filters & SMIRKS transforms |
=======
```bash
npm install
cp .env.example .env.local   # then paste your ANTHROPIC_API_KEY
npm run dev
```

Open http://localhost:3000 in **Chrome** (best voice support). Web Crypto and the mic only work on `localhost` or HTTPS — to try it on a phone over Wi-Fi, use `npm run dev -- --experimental-https`.

**No API key?** The app still works: `/api/chat` falls back to a simple scripted interviewer so you can demo offline. Add the key for the real Claude-powered conversation.

## How it works

- **Voice first.** Hold the mic to talk (Web Speech API), or type. Juno replies out loud (speech synthesis) and as captions. The mute button turns the voice off.
- **Sign-up in the conversation.** Juno asks your name, then a secure keypad pops up for a 6-digit PIN. The PIN never goes into the chat, the mic or the AI.
- **AI interview** (`app/api/chat/route.ts`, model `claude-sonnet-5-5`). Each turn returns structured JSON `{ reply, extracted, section_complete, safety }` validated against the Zod schemas in `lib/schema.ts`. Only the current section's recent turns and draft are sent, never the whole profile. The route stores and logs nothing.
- **Review cards.** After each section Juno shows what it heard; tap to edit, then "Looks good" saves it (encrypted). Leave any time — you'll resume at the next unfinished section.
- **Safety.** Emergency phrases are caught on the device before anything is sent, and Claude flags them too: Juno stops and shows 999 / A&E / NHS 111 (and Samaritans for self-harm).

| File | What it does |
| --- | --- |
| `components/Conversation.tsx` | The whole voice conversation flow |
| `components/Orb.tsx`, `voice-ui.tsx`, `Keypad.tsx`, `ReviewCard.tsx` | UI pieces |
| `lib/voice.ts` | Hold-to-talk speech recognition + text-to-speech |
| `lib/schema.ts` | Profile + section schemas (also Claude's output format) |
| `lib/juno-prompt.ts` | Juno's personality and rules |
| `lib/crypto.ts`, `lib/db.ts`, `lib/vault.tsx` | PBKDF2 + AES-GCM encryption, IndexedDB storage |

Note: Chrome's speech recognition sends audio to Google to transcribe it. Typing avoids that.
>>>>>>> 713f1e5122a13dd7f4239e305d9453ab2d1c3390
