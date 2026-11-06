# A Quantitative Visual Taxonomy of Worldwide Writing Systems

Code for the NeurIPS 2026 paper *A Quantitative Visual Taxonomy of Worldwide
Writing Systems* (Philippe Meyer, Claire Roman).

The pipeline renders the glyphs of 125 historically attested Unicode scripts,
embeds them with a frozen teacher–student encoder, aggregates glyph distances
into script-level distances, selects the script metric and the clustering
method on synthetic benchmarks, and builds a worldwide visual taxonomy. It also
contains the validation experiments of the paper: font sensitivity, k-medoids
sensitivity, recovery of expert script families, real non-Noto glyph images,
robustness to the synthetic generator, and the spatio-temporal analysis.

The glyph encoder is trained in the companion project
[*Contrastive-to-Self-Supervised: A Two-Stage Framework for Script Similarity
Learning*](https://arxiv.org/abs/2603.06180); this repository only uses the
trained checkpoints.

## Installation

Python 3.10, CPU only (no GPU needed).

```bash
conda create -n glyphworld python=3.10
conda activate glyphworld
pip install -r requirements.txt
pip install -e .
```

**Pillow is pinned to 11.1.0 on purpose.** Glyph rendering goes through the
FreeType build bundled with Pillow; Pillow 12 shifts glyph contours by one
pixel, which changes the rendered dataset and, slightly, every downstream
number.

## Data and models

The fonts, the trained checkpoints, and the other input data are archived on
Zenodo: **[DOI to be added]**. Download the archive and extract it at the root
of the repository, so that the paths below exist.

| Path | Content |
|---|---|
| `data/raw/fonts/`, `data/raw/alternative fonts/serif/` | Google Noto fonts (SIL Open Font License) |
| `data/raw/TGSCC-Unicode.txt` | Table of General Standard Chinese Characters (CJK inventory) |
| `data/info/` | Script metadata: locations, time ranges, genealogical family labels |
| `data/baseline_distances/` | Script distances of the baseline encoders (companion project) |
| `data/external/` | External glyph images (real images, held-out Omniglot alphabets), see `data/external/README.md` |
| `models/css/` | Trained teacher and teacher–student checkpoints |

## Running the pipeline

All steps are Python modules, run from the repository root with
`python -m <module>`. They read and write under `data/processed/` and
`results/`. Times are for a laptop CPU (Intel Core i5-1145G7).

| Step | Module | Paper | Time |
|---|---|---|---|
| 1. Render glyphs | `glyphworld.data.build_alphabet_data` | §3.1 | 13 min |
| 2. Glyph embeddings | `glyphworld.models.model_prediction` | §4.1 | 2 min |
| 3. Medoids of large scripts | `glyphworld.clustering.k_medoids` | §4.4 | < 1 min |
| 4. Script distance matrix | `glyphworld.distances.create_global_similarities` | §4.2 | < 1 min |
| 5. Synthetic benchmarks | `glyphworld.synthetic_data.generate_synthetic_dataset` | §3.2 | 1 min |
| 6. Inspect benchmarks | `glyphworld.synthetic_data.inspect_synthetic_dataset` | App. B | < 1 min |
| 7. Metric selection | `glyphworld.synthetic_data.metric_validation --metric-set light` | §4.2, App. C | 2 min |
| 8. Clustering selection | `glyphworld.synthetic_data.synthetic_clustering_validation` | §4.5, App. G | 2 min |
| 9. Worldwide taxonomy | `glyphworld.clustering.clustering` | §5, App. I | < 1 min |
| 10. Knee scan (choice of k) | `glyphworld.clustering.knee_scan` | §5.3, App. I.3 | < 1 min |
| 11. k-medoids sensitivity | `glyphworld.clustering.k_medoids_sensitivity_analysis` | §4.4, App. F | 4 min |
| 12. Font variants | `glyphworld.font_sensitivity.build_font_sensitivity_alphabets` | §4.3, App. D | 1 min |
| 13. Font sensitivity | `glyphworld.font_sensitivity.run_font_sensitivity_analysis` | §4.3, App. D | < 1 min |
| 14. Dataset table | `glyphworld.analysis.excel_to_latex_longtable` | App. A | < 1 min |
| 15. Glyph figure | `glyphworld.visualization.paper_plot_glyphs` | Fig. 1 | < 1 min |
| 16. Family recovery | `glyphworld.analysis.family_recovery` | §5.2, App. J | < 1 min |
| 17. Baseline encoders | `glyphworld.analysis.baseline_family_recovery` | — | < 1 min |
| 18. Held-out Omniglot | `glyphworld.external_validation.omniglot_heldout` | §4.3, App. E | < 1 min |
| 19. Real glyph images | `glyphworld.external_validation.real_images` | §4.3, App. E | < 1 min |
| 20. Spatio-temporal analysis | `glyphworld.analysis.correlation_location_similarity` | §6, App. K | ~2 h |
| 21. Metadata sensitivity | `glyphworld.analysis.spatiotemporal_sensitivity` | §6, App. K.4 | < 1 min |
| 22. Generator robustness | `glyphworld.synthetic_data.synthetic_robustness` | §4.5, App. H | ~2 h |

Steps 1–19 take about 25 minutes; steps 20 and 22 take about 2 hours each and
can run in parallel.


## Citation

```bibtex
@inproceedings{meyer2026worldwide,
  title     = {A Quantitative Visual Taxonomy of Worldwide Writing Systems},
  author    = {Meyer, Philippe and Roman, Claire},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
