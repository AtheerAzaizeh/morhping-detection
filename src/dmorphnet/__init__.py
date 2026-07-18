"""D-MorphNet — face morphing-attack detection pipeline.

Modules:
    config      paths, seeds, dataset targets
    preprocess  CLAHE, standardization, augmentation
    data        manifest loading, identity-disjoint splits, set expansion
    morphgen    landmark morphs + spliced morphs (unseen-generator probe)
    features    backbone feature extraction with labelled caches
    evaluate    classification + MAD metrics (EER, APCER/BPCER, DET, bootstrap CIs)
"""
__version__ = "1.0.0"
