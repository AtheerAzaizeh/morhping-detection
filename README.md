# DMorphNet — Face Morphing Detection

Clean implementation of **DMorphNet** (Gawade et al.): detect whether a face
image is **real** or a **morph** (two identities blended into one) using
**EfficientNet-B6** deep features and an **SVM** classifier.

## Pipeline
```
face → resize 528 + CLAHE → EfficientNet-B6 → 2304-d vector → SVM → Real / Morph
```

## Run
Open **`DMorphNet.ipynb`** in Google Colab (Runtime → GPU) and Run All.
`DEMO = True` runs a fast subset that still produces real results; set
`DEMO = False` for the full-scale run.

- Real faces: FFHQ (256px Kaggle mirror)
- Morphs: landmark morphing (Delaunay + affine warp + seamless clone)
- Reference results (paper): 89.9% accuracy, ROC-AUC 0.965

## Files
- `DMorphNet.ipynb` — the notebook (single, clean, end-to-end)
- `scripts/build_clean_notebook.py` — generator for the notebook
- `archive/` — earlier per-stage notebooks and the full-scale pipeline
