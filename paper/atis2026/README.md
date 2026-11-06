# ATIS 2026 paper — Foraging-Based Frontier Scheduling

Draft submission for **ATIS 2026** (16th Intl. Conf. on Applications and Techniques
in Information Security), Bengaluru, 14–15 Dec 2026.

| | |
|---|---|
| **Track** | 3 — *Artificial Intelligence, Data Security and Cybersecurity* (AI for Cybersecurity / ML for Security) |
| **Format** | Springer **CCIS** (LNCS class), **≤ 12 pages** |
| **Submission** | Microsoft CMT — https://cmt3.research.microsoft.com/ATIS2026 |
| **Deadline** | **7 Nov 2026** (AoE) |
| **Notification** | 30 Nov 2026 · **Camera-ready/registration** 5 Dec 2026 |
| **Proceedings** | Springer CCIS, Scopus-indexed |

## Build

Everything needed is vendored here (`llncs.cls` v2.24 + `splncs04.bst`, taken from the
official ATIS 2026 LaTeX template ZIP). No TeX package installation required. The
bibliography is a manual `thebibliography` (references ordered by first appearance, per
the conference guideline), so **no `bibtex` pass is needed**.

```bash
cd paper/atis2026
pdflatex main            # pass 1
pdflatex main            # pass 2 (resolve references and cross-refs)
# -> main.pdf (12 pages)
```

**Two PDFs are checked in:**
- `main.pdf` — **named** build (`\anonfalse`): camera-ready / your records.
- `main_submission.pdf` — **anonymized** build (`\anontrue`): the one to upload to
  CMT for double-blind review. Regenerate it any time with
  `sed 's/^\\anonfalse/\\anontrue/' main.tex > sub.tex && pdflatex sub && bibtex sub && pdflatex sub && pdflatex sub`.

Or, with latexmk: `latexmk -pdf main.tex`. Also compiles unchanged on Overleaf
(upload the folder; set `main.tex` as the root).

Figures in `figs/` are copied from the repo's `results/` (regenerate with the
experiment drivers in the top-level README).

## ⚠️ Before you submit — two things to decide

1. **Double-blind — use the `\anon` toggle.** ATIS review appears to be
   double-blind. `main.tex` has a one-line switch in the preamble:

   ```latex
   \anonfalse   % named version   (camera-ready / your records) -- DEFAULT
   \anontrue    % anonymized      (upload THIS build to CMT for review)
   ```

   The default is the **named** version (all four authors + RVCE/CSE affiliation).
   **Flip to `\anontrue` and rebuild before uploading the submission PDF to CMT**,
   then flip back to `\anonfalse` for the camera-ready after acceptance.
   Acknowledgments are already commented out and the repo/course name is not cited,
   so `\anontrue` yields a clean double-blind PDF. Authors (named build):
   Manas M. N. (Assoc. Prof.), Chhavi Pareek, Hitarth Mehra (corresponding),
   Chetan M. Patil — Dept. of CSE, R.V. College of Engineering, Bengaluru.

2. **Numbers are drawn from the committed `results/*.csv`.** Spot-check the tables
   against the latest run before submission (abstract sweep: `results/sweep.csv`;
   Tier-B: `results/tierb_*.csv`; ablations: `results/ablation_*.csv`). The headline
   crossover, the constant-hazard collapse, and the single-config robustness table
   are the claims to keep stable.

## Status / TODO

- [x] Full 12-page draft, compiles clean (LNCS, bibliography resolved).
- [ ] Replace anonymized author block for camera-ready.
- [ ] Optional: a code/data availability statement (anonymized repo link) if the
      track requires artifact submission — the CFP FAQ did not specify.
- [ ] Final proofread pass; verify figure legibility at print size.
