## ⚡ Quick Summary

- **Start with a strong, simple baseline:**  
  TF-IDF on `catalog_content` + LightGBM on `log(price)`.

- **Add structured signals:**  
  Use target encoding, per-unit price, and key numeric/text features.

- **Leverage pretrained embeddings:**  
  - Text → Sentence-Transformers (e.g., `all-MiniLM-L6-v2`)  
  - Image → CLIP or EfficientNet  
  Cache all embeddings to disk for speed and reproducibility.

- **Fuse modalities:**  
  Concatenate embeddings + tabular features, then train LightGBM, CatBoost, or a small neural network.

- **Ensemble intelligently:**  
  Combine multiple models (LGBM on TF-IDF, LGBM on embeddings, NN fusion) via out-of-fold (OOF) stacking for robust performance.

- **Stay reproducible and analytical:**  
  Track experiments, use stratified-by-price folds, and perform detailed error analysis to refine predictions.
