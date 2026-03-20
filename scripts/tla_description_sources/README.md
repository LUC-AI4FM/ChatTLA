# TLA+ description harvester

Pulls **official** descriptions for every module in `data/tla-compents-coarse.json` (205 rows) from primary sources — no hand-wavy guesses:

1. **[tlaplus/Examples](https://github.com/tlaplus/Examples)** — `README.md` curated table (spec titles), each folder’s `manifest.json` (**authors**, **paper/PDF URLs**), and the **first `(* … *)` comment block** after `MODULE` in each `.tla` file.
2. **[tlaplus/tlapm](https://github.com/tlaplus/tlapm)** — **`library/*.tla`** for TLAPS proof modules (e.g. `TLAPS`, `NaturalsInduction`, `SequenceTheorems`, `WellFoundedInduction`) that are *not* shipped under `Examples/`.

Each output row records **both** upstream git SHAs so you can diff when the community adds specs.

## Quick start

```bash
./scripts/harvest_tla_descriptions.sh
```

Or:

```bash
python3 scripts/tla_description_sources/harvest_descriptions.py --update-repo -v
python3 scripts/tla_description_sources/audit_descriptions.py   # exit 1 if anything missing
```

First run **clones** into `data/external/` (gitignored — large). The **derived JSON is committed** with pinned commits inside it.

## Outputs

| File | Purpose |
|------|---------|
| `data/derived/tla_descriptions.json` | Per-module: `description`, `confidence`, `provenance`, `paths`, `official_sources`, `upstream.{tlaplus_examples,tlaplus_tlapm}` |
| `data/derived/tla_descriptions_audit.json` | Aggregate stats, both repo commits, refresh commands |

## When upstream changes

```bash
python3 scripts/tla_description_sources/harvest_descriptions.py --update-repo -v
```

Re-run after `git pull` on either upstream; new modules in `tla-compents-coarse.json` are picked up automatically.

## Future extensions

- Add another `(repo_url, path_glob)` pair in `harvest_descriptions.py` for **Lean / Dafny / …** the same way `tlapm` is wired.
- Optionally merge `tla_descriptions.json` back into your component CSV pipeline as a joined column.
