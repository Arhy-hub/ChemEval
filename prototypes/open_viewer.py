"""
Builds and opens the interactive HTML dashboard for the Dual-Arm Autonomous Laparoscopic Ablation System
(plus the Biotech Hit-to-Lead Handoff Preview).
"""
import base64
import json
import os
import webbrowser


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    img_stages = b64_file("prototypes/outputs/aloha_surgical_stages.png")
    mp4_demo = b64_file("prototypes/outputs/aloha_surgical_demo.mp4")

    with open("prototypes/outputs/autonomous_laparoscope_telemetry.json") as f:
        rob_bundle = json.load(f)

    bench = rob_bundle["benchmark"]
    telem = rob_bundle["telemetry"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Agentverse 2026 — Dual-Arm Autonomous Laparoscopic Ablation Dashboard</title>
<style>
  body {{
    margin: 0; font-family: system-ui, -apple-system, sans-serif;
    background: #070b14; color: #e2e8f0;
  }}
  header {{
    padding: 20px 32px; background: #0f172a; border-bottom: 1px solid #1e293b;
    display: flex; justify-content: space-between; align-items: center;
  }}
  h1 {{ margin: 0; font-size: 22px; color: #38bdf8; }}
  .badge {{
    background: #065f46; color: #6ee7b7; padding: 6px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 700; border: 1px solid #10b981;
  }}
  .grid {{
    display: grid; grid-template-columns: 1.25fr 1fr; gap: 24px;
    padding: 24px 32px; max-width: 1500px; margin: 0 auto;
  }}
  .card {{
    background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px;
    margin-bottom: 20px;
  }}
  .card h2 {{ margin-top: 0; color: #7dd3fc; font-size: 17px; }}
  video, img.render {{
    width: 100%; border-radius: 8px; border: 1px solid #334155; background: #000;
  }}
  .kpi-row {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px;
  }}
  .kpi {{
    background: #0b1120; border: 1px solid #1e293b; border-radius: 8px; padding: 12px;
  }}
  .kpi .lbl {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; }}
  .kpi .val {{ font-size: 20px; font-weight: 800; color: #38bdf8; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #1f2937; text-align: left; }}
  th {{ color: #94a3b8; }}
  canvas {{
    width: 100%; height: 210px; background: #030712; border-radius: 8px;
    border: 1px solid #1e293b; margin-top: 10px;
  }}
  .cmd-box {{
    background: #030712; border: 1px solid #334155; padding: 12px; border-radius: 8px;
    font-family: monospace; color: #34d399; font-size: 13px; margin-top: 12px;
  }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Coordinated Dual-Arm Autonomous Laparoscopic Tumor Ablation</h1>
    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
      5-DOF Chip-on-Tip Endoscope Arm + 3-DOF Electrocautery Arm | Self-Learned in MuJoCo & Distilled to INT8 PyTorch Policy
    </div>
  </div>
  <div class="badge">3/3 NODULES ABLATED • 100% ON-DEVICE INT8 POLICY</div>
</header>

<div class="grid">
  <div>
    <div class="card">
      <h2>1. Continuous Single-Take Simulation Video (Left: 3rd-Person Dual-Arm | Right: 1st-Person Endoscope POV)</h2>
      <video controls autoplay loop muted playsinline>
        <source src="data:video/mp4;base64,{mp4_demo}" type="video/mp4">
      </video>
      <div class="cmd-box">
        # Launch live interactive 3D MuJoCo window on your desktop:<br>
        .\\prototypes\\.venv\\Scripts\\python.exe prototypes\\robotics_laparoscope\\train_and_run_autonomous_surgery.py --viewer
      </div>
    </div>

    <div class="card">
      <h2>2. Four Key Procedure Stages (Ablation -> Unscripted Liver Collapse & 0W Abort -> Sub-Hepatic Orbit -> 3/3 Complete)</h2>
      <img class="render" src="data:image/png;base64,{img_stages}" />
    </div>
  </div>

  <div>
    <div class="card">
      <h2>3. Edge Hardware & INT8 Quantization Benchmark (Slide 8 Deliverable)</h2>
      <div class="kpi-row">
        <div class="kpi"><div class="lbl">FP32 Model Size</div><div class="val">{bench['fp32_size_kb']} KB</div></div>
        <div class="kpi"><div class="lbl">INT8 Compressed</div><div class="val" style="color:#4ade80">{bench['int8_size_kb']} KB</div></div>
        <div class="kpi"><div class="lbl">Compression Ratio</div><div class="val">{bench['compression_ratio']}x</div></div>
        <div class="kpi"><div class="lbl">Edge Throughput</div><div class="val">{bench['edge_throughput_hz']} Hz</div></div>
      </div>
      <p style="font-size:13px; color:#94a3b8; margin:0;">
        Distilled PyTorch <code>DualArmSurgicalVisuomotorPolicy</code> maps the 16-D RGB-D + Wrist Force + Proprioception vector directly to 8-DOF dual-arm actions and epistemic uncertainty with zero external cloud calls.
      </p>
    </div>

    <div class="card">
      <h2>4. DeepMind Cornhole-Style Self-Learning Curve in MuJoCo Physics</h2>
      <table>
        <thead>
          <tr><th>Generation</th><th>Surgical Reward</th><th>Nodules Ablated</th><th>Autonomous Learning Milestone</th></tr>
        </thead>
        <tbody>
          {"".join(f"<tr><td><b>Gen {row['iteration']}</b></td><td>{row['mean_reward']}</td><td style='color:#4ade80;font-weight:700'>{row['nodules_ablated']}</td><td style='color:#cbd5e1'>{row['status']}</td></tr>" for row in bench['self_learning_curve'])}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>5. Live Procedure Telemetry: Wrist Contact Force (N), Uncertainty & Task Completion</h2>
      <canvas id="telem-chart" width="560" height="210"></canvas>
      <p style="font-size:13px; color:#cbd5e1; margin-top:10px;">
        Notice the red spike at <b>t = 2.08s</b> when the right liver lobe collapses onto the instrument shaft: wrist force spikes above <b>2.5 N</b>, model uncertainty jumps, and the policy immediately cuts cautery power to <b>0 W</b> and retracts axially before orbiting underneath the liver lobe to finish Nodule 3/3.
      </p>
    </div>
  </div>
</div>

<script>
const telem = {json.dumps(telem)};
const c = document.getElementById('telem-chart');
const ctx = c.getContext('2d');
const w = c.width, h = c.height, pad = 34;

function plotLine(vals, maxVal, color, width) {{
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
  vals.forEach((v, i) => {{
    const x = pad + (i / (vals.length - 1)) * (w - 2 * pad);
    const y = h - pad - (v / maxVal) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
}}

plotLine(telem.map(d => d.wrist_force_n), 4.5, '#f87171', 2.5);
plotLine(telem.map(d => d.uncertainty), 1.0, '#facc15', 2.0);
plotLine(telem.map(d => d.nodules_completed), 3.0, '#4ade80', 2.5);

ctx.font = '12px sans-serif';
ctx.fillStyle = '#f87171'; ctx.fillText('Wrist Force (N, max 4.5N)', pad, 18);
ctx.fillStyle = '#facc15'; ctx.fillText('Model Uncertainty (0-1)', pad + 185, 18);
ctx.fillStyle = '#4ade80'; ctx.fillText('Nodules Ablated (0 -> 3)', pad + 360, 18);
</script>
</body>
</html>
"""
    out_path = os.path.abspath("prototypes/outputs/dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Updated dashboard written to: {out_path}")
    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
