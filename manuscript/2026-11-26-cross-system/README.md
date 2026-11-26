# When does short-window frequency shape improve early warning?

Decision deadlines, measurement noise and representation across simulated power systems. Manuscript snapshot of 26 November 2026.

[Paper (12 pages)](paper.pdf) · [Supplement (25 pages)](supplement.pdf) · [Paper TeX](paper.tex) · [Supplement TeX](supplement.tex)

## Build the manuscripts

Run these commands in this directory with XeLaTeX installed. The three PDF figures are included.

```sh
xelatex -interaction=nonstopmode -halt-on-error paper.tex
xelatex -interaction=nonstopmode -halt-on-error paper.tex
xelatex -interaction=nonstopmode -halt-on-error supplement.tex
xelatex -interaction=nonstopmode -halt-on-error supplement.tex
```

## Companion results

| Files | Contents |
| --- | --- |
| [Recovered bounds](s14-recovered-bounds.csv) | Known and unresolved designs and identification bounds for Supplementary Table S14 |
| [Bootstrap contrasts](r5-bootstrap-contrasts.csv) | Estimates, sampling frames and intervals for the historical IEEE 39 comparisons |
| [Measurement results](measurement-results.csv) · [Historical operating points](historical-operating-points.csv) | Measurement-specific and historical warning results |
| [Aligned records](aligned-834.csv) · [Changed records](aligned-changed-ids.csv) · [Window-level comparison](aligned-window-comparison.csv.gz) | Design- and measurement-matched numerical checks |
| [RK4 readout](rk4-readout.csv) · [Transfer clock](transfer-clock.csv) · [Historical direction counts](historical-direction-units.csv) | Sampling-clock and counting diagnostics |
| [Revision record (Chinese)](r4-review-response.md) | Local review history and the final clarification of the B and B-minus-C intervals |

[SHA256SUMS.txt](SHA256SUMS.txt) records the published manuscripts, figures and companion files. Run `sha256sum -c SHA256SUMS.txt` in this directory before rebuilding the PDFs. The companion tables document selected results. Full simulation replay requires the historical workspaces referenced in the supplement.
