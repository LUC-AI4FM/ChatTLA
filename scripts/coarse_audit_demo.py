"""
coarse_audit_demo.py — One-shot script that executes steps 1-4 of the
coarse-JSON context-recovery plan and emits a self-contained HTML webview.

Steps performed:
  1. Build an index over data/tla-compents-coarse.json with source_path
     resolved against data/FormaLLM/data/.
  2. Build a TF-IDF retriever over (Expressions + StateAndActions +
     StandardLibrary) and precompute top-5 nearest neighbors per row.
  3. (Skipped: wiring into piecewise_gen.py — left for follow-on.)
  4. Run the Diamond gate over a small sample of FormaLLM specs to test
     the headline question: does ANY human-written TLA+ spec pass Diamond?
  5. (Skipped: full benchmark re-eval — multi-hour.)
  6. Render outputs/coarse_audit/webview.html with all results embedded.
"""
from __future__ import annotations

import json
import os
import sys
import glob
import html
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validators.tlc_validator import validate_file  # noqa: E402

COARSE = ROOT / "data" / "tla-compents-coarse.json"
FORMA = ROOT / "data" / "FormaLLM" / "data"
OUTDIR = ROOT / "outputs" / "coarse_audit"
OUTDIR.mkdir(parents=True, exist_ok=True)

# How many specs to actually run TLC+mutation on. Each takes 5-30s.
AUDIT_LIMIT = 10
TLC_TIMEOUT = 25


def rank_tla(p: str) -> tuple:
    """Prefer non-MC, non-_clean as the canonical file."""
    n = os.path.basename(p)
    return (n.startswith("MC"), "_clean" in n, n)


def build_source_index() -> list[dict]:
    """Step 1: attach source_path (canonical) and audit_path (MC variant)."""
    rows = json.loads(COARSE.read_text())
    tla_files = glob.glob(str(FORMA / "**" / "*.tla"), recursive=True)

    canonical: dict[str, str] = {}
    for p in sorted(tla_files, key=rank_tla):
        base = os.path.basename(p).replace(".tla", "").replace("_clean", "")
        canonical.setdefault(base, p)

    # Audit-friendly: a same-named MC* wrapper with a .cfg sibling
    audit_paths: dict[str, tuple[str, str]] = {}
    for p in tla_files:
        n = os.path.basename(p)
        if not n.startswith("MC"):
            continue
        cfg = p.replace("/tla/", "/cfg/").replace(".tla", ".cfg")
        if not os.path.exists(cfg):
            cfg = p.replace(".tla", ".cfg")
        if not os.path.exists(cfg):
            continue
        # The MC wrapper usually extends the same-named base spec
        base = n[2:].replace(".tla", "").replace("_clean", "")
        size = os.path.getsize(p)
        prev = audit_paths.get(base)
        if prev is None or size < os.path.getsize(prev[0]):
            audit_paths[base] = (p, cfg)

    enriched = []
    for r in rows:
        spec = r["Specification"]
        src = canonical.get(spec)
        ap = audit_paths.get(spec)
        r2 = dict(r)
        r2["source_path"] = os.path.relpath(src, ROOT) if src else None
        r2["audit_tla"] = os.path.relpath(ap[0], ROOT) if ap else None
        r2["audit_cfg"] = os.path.relpath(ap[1], ROOT) if ap else None
        enriched.append(r2)
    return enriched


def tokenize_row(r: dict) -> set[str]:
    """Bag of tokens for retrieval (Step 2)."""
    fields = ["Expressions", "StateAndActions", "StandardLibrary",
              "TemporalLogic", "Definitions"]
    toks: set[str] = set()
    for f in fields:
        text = r.get(f, "") or ""
        for chunk in text.split(";"):
            for tok in chunk.replace(",", " ").split():
                tok = tok.strip()
                if tok and len(tok) > 1:
                    toks.add(tok)
    return toks


def build_retriever(rows: list[dict], k: int = 5) -> None:
    """Step 2: precompute top-k nearest neighbors by Jaccard, store in row."""
    bags = [tokenize_row(r) for r in rows]
    for i, bi in enumerate(bags):
        scored = []
        for j, bj in enumerate(bags):
            if i == j or not bi or not bj:
                continue
            inter = len(bi & bj)
            if inter == 0:
                continue
            union = len(bi | bj)
            scored.append((inter / union, j))
        scored.sort(reverse=True)
        rows[i]["neighbors"] = [
            {
                "id": rows[j]["id"],
                "spec": rows[j]["Specification"],
                "score": round(s, 3),
                "source_path": rows[j].get("source_path"),
            }
            for s, j in scored[:k]
        ]


def run_diamond_audit(rows: list[dict]) -> list[dict]:
    """Step 4: run Diamond gate on a small sample of FormaLLM specs."""
    candidates = [r for r in rows if r.get("audit_tla") and r.get("audit_cfg")]
    candidates.sort(key=lambda r: os.path.getsize(ROOT / r["audit_tla"]))
    sample = candidates[:AUDIT_LIMIT]

    results = []
    for r in sample:
        tla = ROOT / r["audit_tla"]
        cfg = ROOT / r["audit_cfg"]
        t0 = time.monotonic()
        # SANY/TLC resolve EXTENDS and -config relative to cwd. The cfg
        # lives in a sibling cfg/ directory, so stage everything in a tmpdir.
        prev_cwd = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="diamond_")
        try:
            for sib in tla.parent.glob("*.tla"):
                shutil.copy2(sib, tmp)
            shutil.copy2(cfg, tmp)
            os.chdir(tmp)
            res = validate_file(Path(tla.name), cfg_path=Path(cfg.name),
                                timeout=TLC_TIMEOUT)
            elapsed = time.monotonic() - t0
            results.append({
                "id": r["id"],
                "spec": r["Specification"],
                "audit_tla": r["audit_tla"],
                "tier": res.tier,
                "is_diamond": res.is_diamond,
                "distinct_states": res.semantic.distinct_states,
                "invariants_checked": res.semantic.invariants_checked,
                "trivial_invariant": res.semantic.trivial_invariant,
                "mutation_tested": res.semantic.mutation_tested,
                "mutation_caught": res.semantic.mutation_caught,
                "runtime_s": round(elapsed, 1),
                "sany_errors": res.sany_errors[:3],
                "violations": res.tlc_violations[:3],
            })
            tag = "DIAMOND" if res.is_diamond else res.tier.upper()
            print(f"  [{tag:8s}] {r['Specification']:30s} "
                  f"states={res.semantic.distinct_states:>6} "
                  f"mut={res.semantic.mutation_caught} "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "id": r["id"], "spec": r["Specification"],
                "audit_tla": r["audit_tla"],
                "tier": "error", "is_diamond": False,
                "error": str(e)[:200], "runtime_s": round(elapsed, 1),
            })
            print(f"  [ERROR   ] {r['Specification']}: {e}")
        finally:
            os.chdir(prev_cwd)
            shutil.rmtree(tmp, ignore_errors=True)
    return results


def render_webview(rows: list[dict], audit: list[dict]) -> str:
    """Step 6: emit a self-contained HTML page with embedded JSON."""
    n_total = len(rows)
    n_with_src = sum(1 for r in rows if r.get("source_path"))
    n_with_audit = sum(1 for r in rows if r.get("audit_tla"))
    n_audited = len(audit)
    n_diamond = sum(1 for a in audit if a.get("is_diamond"))
    n_gold = sum(1 for a in audit if a.get("tier") == "gold")

    rows_json = json.dumps(rows)
    audit_json = json.dumps(audit)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ChatTLA · Coarse-JSON Context Recovery Demo</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
         margin: 24px auto; padding: 0 16px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 32px; padding-top: 12px; border-top: 1px solid #ddd; }}
  .sub {{ color: #666; font-size: 14px; margin-top: 0; }}
  .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
  .stat {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
          padding: 10px 14px; min-width: 130px; }}
  .stat .num {{ font-size: 22px; font-weight: 600; }}
  .stat .lbl {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  .ok {{ color: #1a7f37; }}
  .warn {{ color: #b35900; }}
  .bad {{ color: #cf222e; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border: 1px solid #ddd; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee;
           vertical-align: top; }}
  th {{ background: #f0f0f0; position: sticky; top: 0; }}
  tr:hover {{ background: #f6f8fa; cursor: pointer; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px;
         font-size: 12px; }}
  input[type=text] {{ padding: 8px 12px; width: 360px; font-size: 14px;
                      border: 1px solid #ccc; border-radius: 4px; }}
  .neighbor {{ font-size: 12px; color: #555; }}
  .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
            padding: 16px; margin: 12px 0; }}
  .row-detail {{ display: none; background: #f6f8fa; }}
  .row-detail.open {{ display: table-row; }}
  pre {{ background: #f6f8fa; padding: 10px; overflow-x: auto; font-size: 12px;
         border: 1px solid #ddd; border-radius: 4px; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 10px;
           font-size: 11px; font-weight: 600; }}
  .badge-diamond {{ background: #cfe2ff; color: #084298; }}
  .badge-gold {{ background: #fff3cd; color: #664d03; }}
  .badge-silver {{ background: #e2e3e5; color: #41464b; }}
  .badge-bronze {{ background: #f8d7da; color: #842029; }}
</style>
</head>
<body>

<h1>ChatTLA · Coarse-JSON Context Recovery Demo</h1>
<p class="sub">Steps 1, 2, and 4 of the coarse-context plan, executed end-to-end.
Open <code>outputs/coarse_audit/webview.html</code> in a browser.</p>

<div class="stats">
  <div class="stat"><div class="num">{n_total}</div><div class="lbl">coarse rows</div></div>
  <div class="stat"><div class="num ok">{n_with_src}</div><div class="lbl">with source_path</div></div>
  <div class="stat"><div class="num ok">{n_with_audit}</div><div class="lbl">audit-runnable (MC+cfg)</div></div>
  <div class="stat"><div class="num">{n_audited}</div><div class="lbl">specs audited (sample)</div></div>
  <div class="stat"><div class="num warn">{n_gold}</div><div class="lbl">TLC gold</div></div>
  <div class="stat"><div class="num {'ok' if n_diamond else 'bad'}">{n_diamond}</div><div class="lbl">Diamond pass</div></div>
</div>

<h2>Step 4 · Diamond Audit Results <span class="sub">({n_audited} smallest FormaLLM MC specs)</span></h2>
<div class="panel">
<p style="margin-top:0">Headline question: does <b>any</b> human-written TLA+ spec
in the FormaLLM corpus pass the Diamond gate? The internal RL corpus scored
0/484; this run tests whether the gate is structurally too strict or whether
the failure was specific to model-generated specs.</p>
<table>
<thead><tr>
<th>Spec</th><th>Tier</th><th>States</th><th>Inv ✓</th><th>Trivial?</th>
<th>Mut tested</th><th>Mut caught</th><th>Runtime</th>
</tr></thead>
<tbody id="audit-tbody"></tbody>
</table>
</div>

<h2>Step 1+2 · Index &amp; Retrieval Demo</h2>
<p class="sub">Type a query (operator names, variable names, anything in the
coarse fields). Results ranked by Jaccard over the row's token bag.</p>
<input type="text" id="query" placeholder="e.g. UNCHANGED Paxos, or Bakery, or WF_vars">
<table style="margin-top:12px">
<thead><tr>
<th>id</th><th>Specification</th><th>StandardLibrary</th>
<th>StateAndActions (truncated)</th><th>source_path</th>
</tr></thead>
<tbody id="index-tbody"></tbody>
</table>

<script>
const ROWS = {rows_json};
const AUDIT = {audit_json};

// --- Audit table ---
function renderAudit() {{
  const tb = document.getElementById('audit-tbody');
  if (AUDIT.length === 0) {{
    tb.innerHTML = '<tr><td colspan=8><i>No audit results — re-run scripts/coarse_audit_demo.py</i></td></tr>';
    return;
  }}
  for (const a of AUDIT) {{
    const tr = document.createElement('tr');
    const tier = a.is_diamond ? 'diamond' : (a.tier || 'bronze');
    tr.innerHTML = `
      <td><code>${{a.spec}}</code><br><span class=neighbor>${{a.audit_tla||''}}</span></td>
      <td><span class="badge badge-${{tier}}">${{tier.toUpperCase()}}</span></td>
      <td>${{a.distinct_states ?? '–'}}</td>
      <td>${{a.invariants_checked ?? '–'}}</td>
      <td>${{a.trivial_invariant ? 'yes' : 'no'}}</td>
      <td>${{a.mutation_tested ? 'yes' : 'no'}}</td>
      <td>${{a.mutation_caught ? '<span class=ok>yes</span>' : '<span class=bad>no</span>'}}</td>
      <td>${{a.runtime_s}}s</td>`;
    tb.appendChild(tr);
  }}
}}

// --- Retrieval ---
function tokens(r) {{
  const fields = ['Expressions','StateAndActions','StandardLibrary','TemporalLogic','Definitions'];
  const out = new Set();
  for (const f of fields) {{
    const t = (r[f]||'').replace(/[,;]/g, ' ').split(/\\s+/);
    for (const x of t) if (x.length > 1) out.add(x);
  }}
  return out;
}}
const BAGS = ROWS.map(tokens);

function score(qtoks, i) {{
  const b = BAGS[i];
  let inter = 0;
  for (const t of qtoks) if (b.has(t)) inter++;
  if (inter === 0) return 0;
  return inter / (qtoks.size + b.size - inter);
}}

function renderIndex(query) {{
  const qtoks = new Set((query||'').replace(/[,;]/g,' ').split(/\\s+/).filter(x=>x.length>1));
  let order = ROWS.map((r,i) => ({{i, s: qtoks.size ? score(qtoks, i) : 0}}));
  if (qtoks.size) order = order.filter(o => o.s > 0).sort((a,b)=>b.s-a.s).slice(0,40);
  else order = order.slice(0, 40);

  const tb = document.getElementById('index-tbody');
  tb.innerHTML = '';
  for (const {{i, s}} of order) {{
    const r = ROWS[i];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{r.id}}${{s ? ' <span class=neighbor>('+s.toFixed(2)+')</span>' : ''}}</td>
      <td><b>${{r.Specification}}</b></td>
      <td>${{r.StandardLibrary || '<span class=neighbor>—</span>'}}</td>
      <td><span class=neighbor>${{(r.StateAndActions||'').slice(0,140)}}</span></td>
      <td><code>${{r.source_path||'—'}}</code></td>`;
    const detail = document.createElement('tr');
    detail.className = 'row-detail';
    const nbrs = (r.neighbors||[]).map(n =>
      `<li><b>${{n.spec}}</b> <span class=neighbor>(jaccard=${{n.score}})</span> · <code>${{n.source_path||'—'}}</code></li>`
    ).join('');
    detail.innerHTML = `<td colspan=5><b>Top-5 nearest neighbors (precomputed):</b><ul>${{nbrs}}</ul></td>`;
    tr.onclick = () => detail.classList.toggle('open');
    tb.appendChild(tr);
    tb.appendChild(detail);
  }}
}}

renderAudit();
renderIndex('');
document.getElementById('query').addEventListener('input', e => renderIndex(e.target.value));
</script>
</body>
</html>
"""


def main() -> None:
    print("[1/4] Building source-path index...")
    rows = build_source_index()
    n_src = sum(1 for r in rows if r["source_path"])
    n_aud = sum(1 for r in rows if r["audit_tla"])
    print(f"      {len(rows)} rows, {n_src} with source_path, {n_aud} audit-runnable")

    print("[2/4] Building TF-IDF/Jaccard retriever (top-5 neighbors)...")
    build_retriever(rows, k=5)

    (OUTDIR / "index.json").write_text(json.dumps(rows, indent=2))
    print(f"      wrote {OUTDIR / 'index.json'}")

    print(f"[3/4] Running Diamond audit on {AUDIT_LIMIT} smallest specs "
          f"(timeout={TLC_TIMEOUT}s each)...")
    audit = run_diamond_audit(rows)
    (OUTDIR / "audit.json").write_text(json.dumps(audit, indent=2))
    n_d = sum(1 for a in audit if a.get("is_diamond"))
    print(f"      Diamond pass: {n_d}/{len(audit)}")

    print("[4/4] Rendering self-contained webview...")
    html_path = OUTDIR / "webview.html"
    html_path.write_text(render_webview(rows, audit))
    print(f"      wrote {html_path}")
    print()
    print(f"Open: {html_path}")


if __name__ == "__main__":
    main()
