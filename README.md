# Retail PD Scorecard — Logistic Regression Champion

An end-to-end Probability-of-Default scorecard on retail/personal loans, built to
model-validation standards (SR 11-7, Basel IRB, IFRS 9 ECL, MAS Notice 637). The focus
is **development discipline and validation defensibility** — every modelling decision is
documented as it would be in a Model Development Document (MDD), with the train/test
firewall enforced throughout.

`data prep -> stratified split -> WoE/IV feature engineering -> logistic regression champion
-> validation (AUC/Gini/KS + bootstrap CIs) -> PDO scaling -> calibration -> MLflow artifact lock`

---

## 1. Data

- **Dataset:** Kaggle `credit-risk-dataset` (laotse) — personal/retail loans
- **Size:** 32,581 loans — 25,473 good / 7,108 default
- **Default rate:** 21.8%
- **Target:** `loan_status` (1 = default, 0 = non-default)
- **Features:** age, annual income, home ownership, employment length, loan intent,
  loan grade, loan amount, interest rate, loan-as-%-of-income, historical default flag,
  credit-history length

## 2. Regulatory Framing

| Framework | Relevance |
|-----------|-----------|
| Basel IRB | PD as a risk parameter; rank-ordering + calibration requirements |
| IFRS 9 ECL | Calibrated PDs feed expected credit loss; no probability distortion |
| SR 11-7 | Documentation, effective challenge, reproducible model package |
| MAS Notice 637 | Singapore capital-adequacy context for the standard applied |

## 3. Development Pipeline

**Phase 1 — Data Preparation.** Quality checks surfaced impossible/extreme values
(e.g. `person_age` to 144, `person_emp_length` to 123 years) and informative missingness;
EDA established the treatment plan before any modelling.

**Phase 2 — The Critical Split.** Stratified 80/20 train/test split for out-of-sample
validation; default rate preserved across both partitions (0.218). *This split is the
firewall — all target-using transformations are fit on train only.*

**Phase 3 — Feature Engineering & Selection (train-derived).**
- `loan_int_rate`: median-imputed — null vs. non-null default rates were ~equal
  (0.207 vs. 0.219), so missingness was **non-informative**.
- `person_emp_length`: nulls were statistically significant w.r.t. default, so given their
  own **WoE bin** (missingness as signal).
- Dropped `loan_grade` (high Cramer's V with the historical-default flag; lower IV).
- Dropped `person_age` (low IV; correlated with credit-history length).
- IV used as an **entry gate**, not the sole selection criterion.

**Phase 4 — WoE Binning.** `optbinning` OptimalBinning with monotonic constraints;
all variables confirmed monotonic. Bins fit on train, applied **unchanged** to test —
WoE simultaneously handles outliers, missings, and non-linearity.

**Phase 5 — Modelling.** Unregularised logistic regression via `statsmodels`
`GLM(Binomial)` on WoE features. VIF checked (~1, near-orthogonal after dropping the
correlated raw variables); final feature set locked.

**Phase 6 — Validation (Train & Test).** Discrimination with 1,000-sample bootstrap
95% confidence intervals to show stability, not split-luck.

| Metric | Train | Test | Test 95% CI (bootstrap) |
|--------|-------|------|--------------------------|
| AUC    | 0.876 | 0.866 | (0.855, 0.878) |
| Gini   | 0.752 | 0.732 | (0.709, 0.756) |
| KS     | ~0.59 | ~0.587 | (0.545, 0.625) |

**Phase 7 — Calibration & Scaling (PDO).** Score = `offset + factor * ln(odds)`, with
PDO = 20 (odds double every 20 points), anchored at 600 = 50:1 odds
(`factor = 20/ln 2`, `offset = 600 - factor*ln 50`).
Calibration is **validated, not remediated** — logistic regression is inherently calibrated
under MLE, which the results confirm:

| Partition | Mean predicted PD | Mean actual |
|-----------|-------------------|-------------|
| Train | 0.218155 | 0.218155 |
| Test  | 0.220 | 0.218 |

**Artifact Lock.** Binning object, fitted model, points table, and metrics frozen and
logged to **MLflow** tagged `champion` — the reproducible package a bank hands to its
model-validation function.

## 4. Key Design Decisions

The choices a validator would scrutinise, documented deliberately:

- **Unregularised LR as champion** — interpretability and regulatory acceptance over
  marginal lift; coefficients map directly to a points table.
- **Calibration distinction** — LR calibration is validation evidence (the MLE score
  equation forces mean predicted = mean actual on train); recalibration is reserved for
  non-probability models (e.g. an XGBoost challenger).
- **Missingness as signal** — `person_emp_length` nulls kept as their own bin only after
  *both* a significance check and a bad-rate-direction check; non-informative
  `loan_int_rate` nulls median-imputed.
- **Drop-by-redundancy** — `loan_grade` and `person_age` removed on correlation + IV
  grounds, leaving near-orthogonal WoE features (VIF ~ 1).

## 5. Limitations & Next Steps

- **No out-of-time (OOT) validation** — the dataset lacks origination dates, so only
  out-of-sample (train/test) validation is performed. A time-stamped panel (e.g. a
  mortgage panel dataset) is the proper vehicle for OOT and is planned future work.
- Planned extensions: an **XGBoost challenger** (Platt/isotonic calibration, SHAP,
  DeLong test vs. champion) and **PSI/CSI** monitoring in a deployment context.

## 6. Tech Stack

`pandas` · `numpy` · `statsmodels` (GLM) · `scikit-learn` · `optbinning` · `joblib` ·
`mlflow` · `matplotlib` — developed on Databricks.

## 7. References

- Siddiqi, *Intelligent Credit Scoring* (PDO scaling, scorecard mechanics)
- Baesens et al., *Credit Risk Analytics*
- Hosmer & Lemeshow, *Applied Logistic Regression*
