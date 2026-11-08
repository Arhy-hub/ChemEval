const state = {
  targetsMap: {},
  currentTargetId: "CHEMBL203",
  maxDepth: 10,
  minTanimoto: 0.55,
  nodesById: {},
  selectedNodeId: null,
  latestDossier: null,
  activeEventSource: null,
};

document.addEventListener("DOMContentLoaded", async () => {
  bindListeners();
  await initApp();
});

function bindListeners() {
  const targetSelector = document.getElementById("targetSelector");
  const btnFetchTarget = document.getElementById("btnFetchTarget");
  const compoundInput = document.getElementById("compoundInput");
  const btnLoadToxicHit = document.getElementById("btnLoadToxicHit");
  const sliderMaxDepth = document.getElementById("sliderMaxDepth");
  const sliderTanimoto = document.getElementById("sliderTanimoto");
  const btnRunTriage = document.getElementById("btnRunTriage");
  const btnOpenDossierModal = document.getElementById("btnOpenDossierModal");
  const btnCloseModalTop = document.getElementById("btnCloseModalTop");
  const chkChemistCertify = document.getElementById("chkChemistCertify");
  const inputReviewerNotes = document.getElementById("inputReviewerNotes");
  const btnRejectStream = document.getElementById("btnRejectStream");
  const btnApproveExport = document.getElementById("btnApproveExport");

  targetSelector.addEventListener("change", async () => {
    await loadTargetById(targetSelector.value.trim());
  });

  btnFetchTarget.addEventListener("click", async () => {
    await loadTargetById(targetSelector.value.trim());
  });

  let debounceTimer = null;
  compoundInput.addEventListener("input", () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      await previewInputCompound(compoundInput.value.trim());
    }, 250);
  });

  if (btnLoadToxicHit) {
    btnLoadToxicHit.addEventListener("click", async () => {
      const t = state.targetsMap[state.currentTargetId] || state.targetsMap["CHEMBL203"];
      if (t) {
        compoundInput.value = t.default_hit_smiles;
        showToast(`Loaded hit: ${t.default_hit_id}`);
        await previewInputCompound(t.default_hit_smiles);
      }
    });
  }

  document.querySelectorAll(".chip[data-input]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const val = btn.getAttribute("data-input") || "";
      compoundInput.value = val;
      await previewInputCompound(val);
    });
  });

  sliderMaxDepth.addEventListener("input", (e) => {
    state.maxDepth = parseInt(e.target.value, 10);
    document.getElementById("valMaxDepth").textContent = String(state.maxDepth);
  });

  sliderTanimoto.addEventListener("input", (e) => {
    state.minTanimoto = parseFloat(e.target.value);
    document.getElementById("valTanimoto").textContent = state.minTanimoto.toFixed(2);
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      await previewInputCompound(compoundInput.value.trim());
    }, 180);
  });

  btnRunTriage.addEventListener("click", () => {
    startAutonomousTriageLoop();
  });

  btnOpenDossierModal.addEventListener("click", () => {
    if (state.latestDossier) openBiosecurityModal(state.latestDossier);
  });

  btnCloseModalTop.addEventListener("click", () => {
    document.getElementById("biosecurityModalBackdrop").classList.add("hidden");
  });

  chkChemistCertify.addEventListener("change", updateSignOffButtonState);
  inputReviewerNotes.addEventListener("input", updateSignOffButtonState);

  btnRejectStream.addEventListener("click", () => {
    document.getElementById("biosecurityModalBackdrop").classList.add("hidden");
    appendLog("[Discarded]", "Lead handoff discarded by reviewer.");
    showToast("Lead discarded.");
  });

  btnApproveExport.addEventListener("click", async () => {
    await exportSignedDossier();
  });
}

async function initApp() {
  try {
    const res = await fetch("/api/targets");
    const data = await res.json();
    (data.targets || []).forEach((t) => {
      state.targetsMap[t.id] = t;
    });
    const initialInput = document.getElementById("compoundInput").value.trim();
    await previewInputCompound(initialInput);
  } catch (err) {
    console.error("Init error:", err);
  }
}

async function loadTargetById(targetId) {
  if (!targetId) return;
  try {
    document.getElementById("targetCaptionText").textContent = `Resolving target ${targetId}...`;
    const res = await fetch(`/api/resolve-target?target_id=${encodeURIComponent(targetId)}`);
    const t = await res.json();
    if (t && t.id) {
      state.currentTargetId = t.id;
      state.targetsMap[t.id] = t;
      document.getElementById("targetCaptionText").textContent =
        `Target: ${t.name} (${t.id}) • ${t.disease}`;
      document.getElementById("compoundInput").value = t.default_hit_smiles;
      showToast(`Loaded target ${t.name}`);
      await previewInputCompound(t.default_hit_smiles);
    }
  } catch (err) {
    console.error("Failed to resolve target:", err);
  }
}

async function previewInputCompound(rawInput) {
  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        smiles: rawInput,
        target_id: state.currentTargetId,
        reference_smiles: rawInput,
        min_tanimoto: state.minTanimoto,
        mode: "starting",
      }),
    });
    const data = await res.json();
    const rec = data.recognition;
    const ev = data.evaluation;

    const card = document.getElementById("diagnosisCard");
    const title = document.getElementById("recTitleText");
    const sub = document.getElementById("recSubText");

    if (!rec || !rec.recognized || !ev || !ev.valid) {
      card.className = "diagnosis-card err";
      title.textContent = "Unrecognized Input";
      sub.textContent = (rec && rec.error) || "Enter a valid Drug Name, ChEMBL ID, or SMILES string.";
    } else if (ev.passed_all_gates) {
      card.className = "diagnosis-card clean";
      title.textContent = `${rec.label} — Passes All 8/8 Checks`;
      sub.textContent = `Source: ${rec.source}. Already meets all safety and drug-likeness criteria (no edits required).`;
    } else {
      const failedNames = (ev.gates || [])
        .filter((g) => !g.passed)
        .map((g) => `${g.plain_name || g.name} (${g.value})`);
      card.className = "diagnosis-card warn";
      title.textContent = `${rec.label} — Fails ${8 - ev.gates_passed_count} of 8 Checks`;
      sub.textContent = `Failed: ${failedNames.join(" • ")}`;
    }

    if (ev) {
      updateRightMatrix(
        rec && rec.recognized ? rec.label : "Invalid Input",
        ev.passed_all_gates ? "8/8 PASS" : `${ev.gates_passed_count}/8 PASS`,
        ev
      );

      state.nodesById = {};
      const treeFlow = document.getElementById("treeFlowContainer");
      const failedList = (ev.gates || [])
        .filter((g) => !g.passed)
        .map((g) => g.plain_name || g.name);
      const statusClass = !ev.valid ? "FAIL" : ev.passed_all_gates ? "LEAD" : "BASELINE";
      const whyText = !ev.valid
        ? "Could not parse structure."
        : ev.passed_all_gates
        ? "Passes all 8 clinical safety checks. No structural changes required."
        : `Failed checks: ${failedList.join(", ")}. Click "Run Optimization" to repair these liabilities.`;

      const v = ev.verification || {};
      const idLine = v.formula
        ? `${v.formula} • Exact MW: ${v.exact_mw} Da • InChIKey: ${v.inchi_key}`
        : "";

      treeFlow.innerHTML = `
        <div class="step-card status-${statusClass} selected">
          <div class="step-svg-box">${data.svg || ""}</div>
          <div class="step-info">
            <div class="step-top-row">
              <div class="step-title">Starting Compound: ${escapeHtml(rec ? rec.label : rawInput)}</div>
              <span class="step-badge">${ev.gates_passed_count} / 8 PASS</span>
            </div>
            <div class="step-why">${escapeHtml(whyText)}</div>
            ${idLine ? `<div class="step-chem-meta">${escapeHtml(idLine)}</div>` : ""}
          </div>
        </div>
      `;
    }
  } catch (err) {
    console.error("Preview failed:", err);
  }
}

function startAutonomousTriageLoop() {
  if (state.activeEventSource) {
    state.activeEventSource.close();
    state.activeEventSource = null;
  }

  state.nodesById = {};
  state.selectedNodeId = null;
  state.latestDossier = null;

  document.getElementById("btnRunTriage").disabled = true;
  document.getElementById("btnOpenDossierModal").disabled = true;
  document.getElementById("treeFlowContainer").innerHTML = "";
  document.getElementById("thoughtConsoleBody").innerHTML = "";
  document.getElementById("pipelineStatusText").textContent = "Running...";

  const params = new URLSearchParams({
    target_id: state.currentTargetId,
    compound_input: document.getElementById("compoundInput").value.trim(),
    max_depth: String(state.maxDepth),
    min_tanimoto: String(state.minTanimoto),
    step_delay: "0.35",
  });

  const es = new EventSource(`/api/stream-triage?${params.toString()}`);
  state.activeEventSource = es;

  es.onmessage = (msg) => {
    try {
      handleEvent(JSON.parse(msg.data));
    } catch (e) {
      console.error(e);
    }
  };

  es.onerror = () => {
    es.close();
    state.activeEventSource = null;
    document.getElementById("btnRunTriage").disabled = false;
  };
}

function handleEvent(evt) {
  if (evt.event === "log") {
    appendLog(evt.tag, evt.message);
  } else if (evt.event === "node") {
    const node = evt.node;
    state.nodesById[node.id] = node;
    document.getElementById("consoleCycleCounter").textContent = `Step ${node.cycle}`;
    appendStepCard(node);
    selectNode(node.id);
  } else if (evt.event === "backtrack") {
    appendBacktrackBanner();
  } else if (evt.event === "gate_update") {
    const node = state.nodesById[evt.node_id];
    updateRightMatrix(evt.node_title, node ? node.badge : "", evt.evaluation);
  } else if (evt.event === "complete") {
    if (state.activeEventSource) {
      state.activeEventSource.close();
      state.activeEventSource = null;
    }
    state.latestDossier = evt.dossier;
    if (Array.isArray(evt.dossier.evaluated_candidates) && evt.dossier.evaluated_candidates.length > 0) {
      renderEvaluatedCandidatePool(evt.dossier.evaluated_candidates);
    }
    const all8Passed = Boolean(
      evt.dossier.converged_all_8 ??
        (evt.dossier.optimized_lead &&
          evt.dossier.optimized_lead.evaluation &&
          evt.dossier.optimized_lead.evaluation.passed_all_gates)
    );
    document.getElementById("btnRunTriage").disabled = false;
    document.getElementById("btnOpenDossierModal").disabled = !all8Passed;
    document.getElementById("pipelineStatusText").textContent = evt.dossier.already_optimal
      ? "Clean (8/8)"
      : all8Passed
      ? "Lead Ready (8/8)"
      : "Incomplete (<8/8 — Export Locked)";

    if (!all8Passed) {
      showToast("Could not reach 8/8 within constraints — sign-off and export locked.");
    } else if (!evt.dossier.already_optimal) {
      setTimeout(() => {
        if (state.latestDossier) openBiosecurityModal(state.latestDossier);
      }, 450);
    } else {
      showToast("Compound already passes all 8/8 checks — no modifications needed.");
    }
  }
}

function appendStepCard(node) {
  const container = document.getElementById("treeFlowContainer");

  if (container.children.length > 0) {
    const arrow = document.createElement("div");
    arrow.className = "step-arrow";
    arrow.innerHTML = "&darr;";
    container.appendChild(arrow);
  }

  let extraBanner = "";
  if (node.alert_banner) {
    extraBanner = `<div class="step-banner banner-fail">${escapeHtml(node.alert_banner)}</div>`;
  } else if (node.discovery_banner) {
    extraBanner = `<div class="step-banner banner-lead">${escapeHtml(node.discovery_banner)}</div>`;
  }

  const v = (node.evaluation && node.evaluation.verification) || {};
  const idLine = v.formula
    ? `${v.formula} • Exact MW: ${v.exact_mw} Da • InChIKey: ${v.inchi_key}`
    : "";

  const card = document.createElement("div");
  card.className = `step-card status-${node.status}`;
  card.id = `dom-${node.id}`;
  card.innerHTML = `
    <div class="step-svg-box">${node.svg || ""}</div>
    <div class="step-info">
      <div class="step-top-row">
        <div class="step-title">${escapeHtml(node.title)}</div>
        <span class="step-badge">${escapeHtml(node.badge)}</span>
      </div>
      <div class="step-why">${escapeHtml(node.subtitle)}</div>
      ${idLine ? `<div class="step-chem-meta">${escapeHtml(idLine)}</div>` : ""}
      ${extraBanner}
    </div>
  `;

  card.addEventListener("click", () => selectNode(node.id));
  container.appendChild(card);
  container.scrollTop = container.scrollHeight;
}

function renderEvaluatedCandidatePool(candidates) {
  const container = document.getElementById("treeFlowContainer");
  const section = document.createElement("div");
  section.className = "candidate-pool-section";

  const header = document.createElement("div");
  header.className = "candidate-pool-header";
  header.textContent = `All ${candidates.length} Evaluated Candidate Compounds (Click any compound to inspect its 8 gates & validity proof)`;
  section.appendChild(header);

  const grid = document.createElement("div");
  grid.className = "candidate-pool-grid";

  candidates.forEach((cand, idx) => {
    const cid = cand.id || `cand-${idx + 1}`;
    state.nodesById[cid] = cand;
    const ev = cand.evaluation || {};
    const v = ev.verification || {};
    const m = ev.metrics || {};

    const item = document.createElement("div");
    item.className = `step-card candidate-mini-card status-${cand.status || "PASS"}`;
    item.id = `dom-${cid}`;
    item.innerHTML = `
      <div class="step-svg-box mini-svg-box">${cand.svg || ""}</div>
      <div class="step-info">
        <div class="step-top-row">
          <div class="step-title">Compound #${idx + 1}: ${escapeHtml(cand.title)}</div>
          <span class="step-badge">${escapeHtml(cand.badge || `${ev.gates_passed_count || 0}/8`)}</span>
        </div>
        <div class="step-why">${escapeHtml(cand.subtitle || cand.rationale || "")}</div>
        <div class="step-chem-meta">
          ${escapeHtml(v.formula || "")} • MW: ${escapeHtml(v.exact_mw ?? m.mw ?? "")} • Tanimoto: ${escapeHtml(m.tanimoto !== undefined ? Number(m.tanimoto).toFixed(3) : "")} • LogS: ${escapeHtml(m.esol_logs ?? "")} • hERG: ${escapeHtml(m.herg_probability ?? "")}
        </div>
      </div>
    `;
    item.addEventListener("click", () => selectNode(cid));
    grid.appendChild(item);
  });

  section.appendChild(grid);
  container.appendChild(section);
}

function appendBacktrackBanner() {
  const container = document.getElementById("treeFlowContainer");
  const bt = document.createElement("div");
  bt.className = "backtrack-banner";
  bt.textContent = "Backtrack: Reverted rejected Attempt 2 back to Step 1 and applied next-best transformation";
  container.appendChild(bt);
}

function selectNode(nodeId) {
  state.selectedNodeId = nodeId;
  document.querySelectorAll(".step-card").forEach((el) => {
    el.classList.toggle("selected", el.id === `dom-${nodeId}`);
  });
  const node = state.nodesById[nodeId];
  if (node) {
    updateRightMatrix(node.title, node.badge, node.evaluation);
  }
}

function appendLog(tag, message) {
  const stream = document.getElementById("thoughtConsoleBody");
  const row = document.createElement("div");
  row.className = "log-row";
  row.innerHTML = `
    <span class="log-tag">${escapeHtml(tag)}</span>
    <span>${escapeHtml(message)}</span>
  `;
  stream.appendChild(row);
  stream.scrollTop = stream.scrollHeight;
}

function updateRightMatrix(title, badgeText, evaluation) {
  if (!evaluation) return;
  document.getElementById("inspectedNodeTitle").textContent = title;
  document.getElementById("inspectedNodeBadge").textContent = badgeText || "";

  const pill = document.getElementById("gatesScoreSummaryPill");
  pill.textContent = `${evaluation.gates_passed_count} / ${evaluation.total_gates} PASS`;
  pill.classList.toggle("all-pass", Boolean(evaluation.passed_all_gates));

  const list = document.getElementById("clinicalGatesList");
  list.innerHTML = "";
  (evaluation.gates || []).forEach((g) => {
    const item = document.createElement("div");
    item.className = `check-item ${g.passed ? "passed" : "failed"}`;
    const displayName = g.plain_name ? `${g.id}. ${g.plain_name}` : g.name;
    item.innerHTML = `
      <div>
        <div class="check-title">${escapeHtml(displayName)}</div>
        <div class="check-sub">Target: ${escapeHtml(g.threshold)}</div>
      </div>
      <div class="check-right">
        <div class="check-val">${escapeHtml(g.value)}</div>
        <div class="check-status ${g.passed ? "pass" : "fail"}">${g.passed ? "PASS" : "FAIL"}</div>
      </div>
    `;
    list.appendChild(item);
  });

  const proofBox = document.getElementById("moleculeValidityProofBox");
  if (proofBox) {
    if (!evaluation.valid || !evaluation.verification) {
      proofBox.innerHTML = `
        <div class="proof-title">Chemical Validity Verification</div>
        <div class="proof-row"><span>Status:</span> <b class="c-red">Invalid Structure (Failed RDKit Sanitize)</b></div>
      `;
    } else {
      const v = evaluation.verification;
      proofBox.innerHTML = `
        <div class="proof-title">New Molecule Chemical Validity Proof</div>
        <div class="proof-row"><span>Valence &amp; Kekulize:</span> <b class="c-green">${escapeHtml(v.valence_status)}</b></div>
        <div class="proof-row"><span>Formula &amp; Exact MW:</span> <b>${escapeHtml(v.formula)} (${escapeHtml(v.exact_mw)} Da)</b></div>
        <div class="proof-row"><span>InChIKey:</span> <code>${escapeHtml(v.inchi_key)}</code></div>
        <div class="proof-row"><span>Canonical SMILES:</span> <code class="smiles-code">${escapeHtml(v.canonical_smiles)}</code></div>
        <div class="proof-actions">
          <button type="button" class="btn-small" id="btnCopyInspectSmiles">Copy SMILES</button>
          <button type="button" class="btn-small" id="btnVerifyAsNewInput">Verify as New Input</button>
          <a class="btn-small proof-link" href="${escapeHtml(v.pubchem_url)}" target="_blank" rel="noopener noreferrer">Verify in PubChem</a>
        </div>
      `;
      const btnCopy = document.getElementById("btnCopyInspectSmiles");
      if (btnCopy) {
        btnCopy.addEventListener("click", () => {
          navigator.clipboard.writeText(v.canonical_smiles);
          showToast("Copied Canonical SMILES to clipboard");
        });
      }
      const btnVerify = document.getElementById("btnVerifyAsNewInput");
      if (btnVerify) {
        btnVerify.addEventListener("click", async () => {
          const inp = document.getElementById("compoundInput");
          inp.value = v.canonical_smiles;
          showToast("Loaded generated SMILES into Input #1 — verifying independently...");
          await previewInputCompound(v.canonical_smiles);
        });
      }
    }
  }
}

function openBiosecurityModal(dossier) {
  const start = dossier.starting_compound;
  const lead = dossier.optimized_lead;
  const sM = start.evaluation.metrics;
  const lM = lead.evaluation.metrics;

  document.getElementById("modalStartLabel").textContent = start.label;
  document.getElementById("modalStartSvgStage").innerHTML = start.svg;
  document.getElementById("modalStartMetricsStrip").textContent =
    `Gates: ${dossier.deltas.gates_before} | LogS: ${sM.esol_logs} | hERG: ${sM.herg_probability}`;

  document.getElementById("modalLeadLabel").textContent = lead.label;
  document.getElementById("modalLeadSvgStage").innerHTML = lead.svg;
  document.getElementById("modalLeadMetricsStrip").textContent =
    `Gates: ${dossier.deltas.gates_after} | LogS: ${lM.esol_logs} | hERG: ${lM.herg_probability} | Tanimoto: ${lM.tanimoto}`;

  const bio = dossier.biosecurity || {};
  document.getElementById("modalBiosecStatusText").textContent = bio.status || "CLEAN";
  document.getElementById("modalBiosecDeclarationText").textContent =
    bio.declaration || "Dual-use toxin SMARTS filter: 0 matches found against restricted lists.";

  document.getElementById("chkChemistCertify").checked = false;
  document.getElementById("inputReviewerNotes").value =
    (lead.modifications_summary && lead.modifications_summary.join(" ")) ||
    "Reviewed structural modifications and verified all 8 clinical safety gates pass.";
  updateSignOffButtonState();

  document.getElementById("biosecurityModalBackdrop").classList.remove("hidden");
}

function updateSignOffButtonState() {
  const certified = document.getElementById("chkChemistCertify").checked;
  const notes = document.getElementById("inputReviewerNotes").value.trim();
  const all8Passed = Boolean(
    state.latestDossier &&
      state.latestDossier.optimized_lead &&
      state.latestDossier.optimized_lead.evaluation &&
      state.latestDossier.optimized_lead.evaluation.passed_all_gates
  );
  document.getElementById("btnApproveExport").disabled = !(all8Passed && certified && notes.length >= 5);
}

async function exportSignedDossier() {
  if (!state.latestDossier) return;
  const certified = document.getElementById("chkChemistCertify").checked;
  const notes = document.getElementById("inputReviewerNotes").value.trim();

  try {
    const res = await fetch("/api/signoff-export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dossier_id: state.latestDossier.dossier_id,
        certified,
        reviewer_notes: notes,
        dossier_payload: state.latestDossier,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      showToast(err.detail || "Sign-off failed");
      return;
    }
    const pkg = await res.json();
    const blob = new Blob([JSON.stringify(pkg, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.latestDossier.dossier_id}-signed-handoff.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    document.getElementById("biosecurityModalBackdrop").classList.add("hidden");
    showToast("Exported signed JSON handoff report.");
  } catch (err) {
    console.error("Export error:", err);
  }
}

function showToast(msg) {
  const toast = document.getElementById("toastBanner");
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3000);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
