# TLA+ description harvester

Pulls **official** descriptions for every module in `data/tla-compents-coarse.json` (205 rows) from primary sources — no hand-wavy guesses:

1. **[tlaplus/Examples](https://github.com/tlaplus/Examples)** — `README.md` curated table (spec titles), each folder’s `manifest.json` (**authors**, **paper/PDF URLs**), and the **first `(* … *)` comment block** after `MODULE` in each `.tla` file. For **`.pdf`** links in `sources`, the harvester **downloads the PDF** and extracts **text from the first pages** with **`pypdf`** (cached under `data/derived/.pdf_cache/`), then appends that excerpt to the description so the model sees prose, not only a bare URL.
2. **[tlaplus/tlapm](https://github.com/tlaplus/tlapm)** — **`library/*.tla`** for TLAPS proof modules (e.g. `TLAPS`, `NaturalsInduction`, `SequenceTheorems`, `WellFoundedInduction`) that are *not* shipped under `Examples/`.

Each output row records **both** upstream git SHAs so you can diff when the community adds specs.

### Dataset shape (regeneration-oriented)

Each row is built for feeding an LM that **reconstructs** the spec:

- `id`: `<module_name_lowercase>_001`
- `module_name`: exact TLA+ module name
- `coarse_id`: index from `data/tla-compents-coarse.json`
- `description`: **`{ "narrative", "technical" }`** — paper-style abstract + structured reconstruction fields (variables, init, actions, fairness, invariants, design decisions). See `structured_dataset.py` for the full schema string passed to the LLM.

**Without `--llm` (default — SANY toolchain):** Runs **`java -cp tla2tools.jar tla2sany.xml.XMLExporter`** on each `.tla` file, parses the resulting XML AST, and fills **`description.technical`** (constants, variables with SANY-assigned levels, Init/Next/Spec bodies reconstructed from the AST, action-level operators as structured actions, invariants, temporal properties, fairness tokens). Falls back to regex-based `tla_static_extract.py` for the ~20% of specs where SANY can't resolve module dependencies. **`--no-static-extract`** disables both (empty technical shell unless you use `--llm`).

**With `--llm`:** Ollama fills `description` using the formal prompt in `structured_dataset.py` (set `OLLAMA_HOST`, optional `TLA_DATASET_LLM_MODEL`, default `gpt-oss:20b`). Overrides SANY/static extraction for the same schema.

## Quick start

```bash
./scripts/harvest_tla_descriptions.sh
```

Or:

```bash
python3 scripts/tla_description_sources/harvest_descriptions.py --update-repo -v
python3 scripts/tla_description_sources/audit_descriptions.py   # exit 1 if anything missing

# Full structured descriptions (slow; needs Ollama):
# python3 scripts/tla_description_sources/harvest_descriptions.py --update-repo --llm -v
```

Use `--skip-pdf` for a fast/offline run (skips PDF download and text extraction; manifest still lists PDF URLs in `References:`).

First run **clones** into `data/external/` (gitignored — large). The **derived JSON is committed** with pinned commits inside it.

## Outputs

| File | Purpose |
|------|---------|
| `data/derived/tla_descriptions.json` | Per-module: `id`, `module_name`, `coarse_id`, structured `description`, `confidence`, `provenance`, `paths`, optional `official_sources` / `authors`, optional `pdf_excerpt_meta`, optional `llm` when `--llm` was used |
| `data/derived/tla_descriptions_audit.json` | Aggregate stats, both repo commits, refresh commands |

## When upstream changes

```bash
python3 scripts/tla_description_sources/harvest_descriptions.py --update-repo -v
```

Re-run after `git pull` on either upstream; new modules in `tla-compents-coarse.json` are picked up automatically.

## Future extensions

- Add another `(repo_url, path_glob)` pair in `harvest_descriptions.py` for **Lean / Dafny / …** the same way `tlapm` is wired.
- Optionally merge `tla_descriptions.json` back into your component CSV pipeline as a joined column.
