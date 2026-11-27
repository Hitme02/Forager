# ATIS 2026 paper — Foraging-Based Frontier Scheduling

**Accepted** at **ATIS 2026** (16th Intl. Conf. on Applications and Techniques in
Information Security), CMR University, Bengaluru, 14–15 Dec 2026 — paper 452. This folder
holds the **camera-ready** version (named build, 12 pages).

| | |
|---|---|
| **Track** | 3 — *Artificial Intelligence, Data Security and Cybersecurity* |
| **Format** | Springer **CCIS** (LNCS class), **≤ 12 pages** |
| **Registration and camera-ready** | 5 Dec 2026, 23:59 IST |
| **Proceedings** | Springer CCIS, Scopus-indexed |

## Build

`llncs.cls` (v2.24) is vendored from the official template. The bibliography is a manual
`thebibliography`, ordered by first citation, so no `bibtex` pass is needed.

```bash
cd paper/atis2026
pdflatex main && pdflatex main     # -> main.pdf (12 pages)
```

`latexmk -pdf main.tex` also works, and the folder compiles unchanged on Overleaf.
`\anontrue` in the preamble gives the anonymised build used for review.

## Generated content

The figure (`figs/fig_speedup.pdf`, vector) and the table bodies (`tables/*.tex`) are
generated from `results/cr/`; nothing in them is typed by hand:

```bash
python -m ffs.experiments.plot_cr     # -> figs/fig_speedup.pdf
python -m ffs.experiments.tables_cr   # -> tables/{sweep,robust,llm}.tex
```

The drivers that produce `results/cr/` are listed under *Camera-ready experiments* in the
top-level README.
