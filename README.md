# WIUT Hackathon EDA Website


**Team ID:** `8BA88A02`  
**Task:** Financial Alert Escalation Prediction  
**Evaluation Metric:** ROC-AUC  
**Validated Out-Of-Fold CV:** `0.6505` ROC-AUC  

---

### Dear Reviewers and Organizing Committee,

We are pleased to submit our solution for the **WIUT Hackathon Elimination Stage**. Below is an overview of our methodology, deliverables, and reproducibility verification.

---

### Deliverables Summary

1. **Prediction File (`team_8BA88A02.csv`):**
   - Exactly 6,000 predictions for the hidden test set.
   - Required schema (`signal_id,ehtimollik`), strictly zero NaN/null values.
   - Continuous, calibrated probabilities within `[0.2216, 0.7193]` (mean: `0.4786`).

2. **Reproducible Jupyter Notebook (`hackathon_notebook.ipynb`):**
   - End-to-end reproducible pipeline from raw data loading to final submission generation.
   - Pre-rendered with all exploratory data analysis visualizations, cross-validation metrics, and prediction distributions.
   - Verifiably generates the exact predictions present in `team_8BA88A02.csv` bit-for-bit.

3. **Public Interactive EDA Website:**
   - **Live URL:** `[[URL](https://feruz-bb.github.io/WIUT_hackathon_EDA_website/)]`
   - Features 12 interactive Chart.js visualizations, behavioral pattern breakdowns, and our 5-stage iterative experiment progression.

---

### Technical Highlights & Modeling Rationale

Through deep exploratory data analysis on 10 million transactions across 20,000 signals, we identified critical behavioral signatures that shaped our winning architecture:

1. **Noise Elimination via Top-75 Feature Selection:**
   - Rather than blinding models with hundreds of weak variables, we engineered 246 multi-dimensional features (direction flows, pre-signal acceleration, sequence switches, and histogram distribution bins) and applied permutation and split-importance selection to isolate the **Top-75 high-impact features**. This alone lifted single-model CV by **+0.0036 AUC**.

2. **Overfitting Mitigation (Shallow Trees & Heavy Regularization):**
   - 200 Bayesian optimization trials (Optuna) proved that complex, deep trees severely overfit noisy financial surveillance signals.
   - Optimal architectures converged to shallow structures:
     - **LightGBM:** `max_depth = 3`, `min_child_samples = 268`, `reg_lambda = 21.3`
     - **CatBoost:** `depth = 2`, `l2_leaf_reg = 0.45`
     - **XGBoost:** `max_depth = 2`, `min_child_weight = 144`, `reg_lambda = 20.5`

3. **15-Model Multi-Seed Bagging (`0.6505` ROC-AUC):**
   - Ensembled 3 diverse model families across 5 independent seeds (42, 2024, 777, 1337, 12345) using 5-fold Stratified Cross-Validation.
   - Performance-weighted blending smoothed out fold-level variance and delivered our peak cross-validation score of **0.6505 ROC-AUC**.

4. **100% Leak-Free Validation:**
   - Out-of-fold predictions were strictly recorded without target leakage. Class imbalance (17.2%) was handled algorithmically via loss weighting (`is_unbalance=True`, `scale_pos_weight=4.82`) to preserve genuine transaction density.

---

Thank you for your time and evaluation!

*Team 8BA88A02*
