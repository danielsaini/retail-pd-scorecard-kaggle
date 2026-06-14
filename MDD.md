# Model Development Document (MDD)

**Model:** Retail Probability-of-Default (PD) Scorecard
**Model type:** Application credit scorecard — logistic regression (WoE)
**Champion / Challenger:** Champion (logistic regression); XGBoost challenger planned
**Version:** 1.0
**Author:** Daniel Cherkasskiy Saini
**Date:** 14-06-2026
**Status:** Development — pending independent validation

---

## 1. Executive Summary

This document describes the development of a retail PD scorecard estimating the
probability of default on personal loans. The champion model is an unregularised
logistic regression fitted on Weight-of-Evidence (WoE) transformed features, selected
for interpretability and regulatory acceptability.

On a held-out 20% test sample the model achieves **AUC 0.866 / Gini 0.732 / KS ~0.59**,
with 1,000-sample bootstrap 95% confidence intervals confirming the metrics are stable
rather than split-specific. The model is well calibrated by construction: mean predicted
PD equals the observed default rate on the training sample (0.2182 = 0.2182).

The principal limitation is the absence of an out-of-time (OOT) validation window, as the
development dataset carries no origination dates. This is documented in Section 10.

## 2. Purpose, Scope & Intended Use

The model estimates the one-period probability of default for retail personal-loan
applicants, to support credit-approval decisioning and to provide a calibrated PD input
to IFRS 9 expected-credit-loss estimation.

**In scope:** retail personal loans represented in the development population.
**Out of scope / not to be used for:** corporate or secured-lending portfolios, segments
not represented in the development data, or any use requiring a lifetime (multi-period) PD
without further extension.

## 3. Regulatory & Business Context

| Framework | Relevance |
|-----------|-----------|
| Basel IRB | PD as a risk parameter; rank-ordering and calibration requirements |
| IFRS 9 ECL | Calibrated PDs feed expected credit loss; predicted probabilities must not be distorted |
| SR 11-7 | Conceptual soundness, ongoing monitoring, outcomes analysis; documentation and effective challenge |
| MAS Notice 637 | Singapore capital-adequacy context for the modelling standard applied |

## 4. Data

**Source.** Public retail credit-risk dataset (Kaggle, laotse `credit-risk-dataset`).

**Population & target.**
- Observations: 32,581 loans (25,473 non-default, 7,108 default).
- Observed default rate: 21.8%.
- Target: `loan_status` (1 = default, 0 = non-default).
- *Note:* in a production setting the default definition would follow the regulatory
  90-days-past-due standard; here the supplied binary flag is taken as given.

**Candidate features.** Applicant age, annual income, home ownership, employment length,
loan intent, loan grade, loan amount, interest rate, loan-as-%-of-income, historical
default flag, credit-history length.

**Data quality.** Sanity checks surfaced implausible values requiring treatment, e.g.
`person_age` up to 144 and `person_emp_length` up to 123 years; these are absorbed by WoE
binning rather than hard-deleted. Missingness was assessed for informativeness (Section 7).

**Limitation.** No origination/observation dates are present, precluding a time-based OOT
split (see Section 10).

## 5. Sampling & Representativeness

The data was partitioned into an 80% training and 20% test sample using a **stratified**
split on the target, preserving the default rate across both partitions (0.218). The split
is the organising firewall of the development: **all target-using transformations (WoE, IV,
imputation values) are fitted on the training sample only and applied unchanged to the test
sample**, preventing leakage that would inflate apparent performance.

## 6. Methodology & Conceptual Soundness

The champion is an **unregularised logistic regression on WoE-transformed features**,
chosen deliberately over higher-capacity alternatives:

- **Interpretability** — each coefficient maps directly to a points contribution in a
  scorecard, supporting adverse-action explanation and validator review.
- **Calibration by construction** — fitted by maximum likelihood, the score equation forces
  mean predicted PD to equal the observed default rate, so the output is a genuine
  probability suitable for IFRS 9 ECL without a recalibration layer.
- **Monotonicity & stability** — WoE binning with monotonic constraints enforces
  economically sensible, defensible relationships and linearises the predictors for the
  logit, improving stability.

A more flexible model (e.g. XGBoost) is reserved as a *challenger* for benchmarking, where
its probabilities would require post-hoc Platt/isotonic calibration; it is not proposed as
the champion on interpretability and regulatory-acceptance grounds.

## 7. Variable Selection & Treatment

Selection proceeded from a long list to a final set, with Information Value (IV) used as an
**entry gate** rather than the sole criterion; multivariate checks (correlation, VIF,
coefficient signs) governed the final set.

**Missingness treatment (decided on train):**
- `loan_int_rate` — **median-imputed**. Default rates for missing vs. non-missing were
  approximately equal (0.207 vs. 0.219), indicating non-informative missingness.
- `person_emp_length` — **own WoE bin**. Missingness was statistically significant with
  respect to default, i.e. informative; isolating it preserves the signal.

**Redundancy-based drops:**
- `loan_grade` — dropped; high categorical association (Cramér's V) with the historical
  default flag and lower IV; retaining both risked unstable coefficients.
- `person_age` — dropped; low IV and high correlation with credit-history length.

**Per-variable IV summary:** 

| name | dtype | iv |
|------|-------|------|
| person_income | numerical | 0.5817 |
| person_home_ownership | categorical | 0.3841 |
| person_emp_length | numerical | 0.0708 |
| loan_intent | categorical | 0.0941 |
| loan_amnt | numerical | 0.0818 |
| loan_int_rate | numerical | 0.6920 |
| loan_percent_income | numerical | 0.9542 |
| cb_person_default_on_file | categorical | 0.1633 |


## 8. Feature Engineering — WoE Binning

Optimal binning was performed with `optbinning` under monotonic constraints; all retained
variables were confirmed monotonic. WoE replaces each raw value with the log-odds of its
bin, simultaneously handling outliers, missing values, and non-linearity. Bins were fitted
on the training sample and applied unchanged to the test sample.

**Final binning / WoE tables:** 
See appendice B.


## 9. Model Estimation

The champion was estimated using `statsmodels` `GLM(family=Binomial)` — an unregularised
logistic regression — on the WoE features.

- **Multicollinearity:** VIF on the WoE features was ~1, confirming near-orthogonality
  after the redundancy-based drops.
- **Coefficient signs:** all coefficients reviewed for economic sense (consistent with the
  WoE direction). 

**Final coefficient table:** 

## 10. Performance Testing (Developer Validation)

### 10.1 Discrimination

| Metric | Train | Test | Test 95% CI (1,000-sample bootstrap) |
|--------|-------|------|---------------------------------------|
| AUC    | 0.876 | 0.866 | (0.855, 0.878) |
| Gini   | 0.752 | 0.732 | (0.709, 0.756) |
| KS     | ~0.59 | ~0.587 | (0.545, 0.625) |

The narrow gap between train and test, and confidence intervals that exclude weak
performance, indicate the model discriminates well and does not overfit.

### 10.2 Calibration

Logistic regression is calibrated by construction; the development results confirm it:

| Partition | Mean predicted PD | Mean actual |
|-----------|-------------------|-------------|
| Train | 0.218155 | 0.218155 |
| Test  | 0.220 | 0.218 |

The decile calibration table [INSERT / reference] shows predicted vs. observed default
rates aligning across score bands. Calibration here is **validation evidence**, not a
remediation step.

## 11. Score Scaling (PDO)

Probabilities were scaled to an interpretable points system using the
Points-to-Double-the-Odds framework (Siddiqi):

`Score = offset + factor * ln(odds)`, with PDO = 20 (odds double every 20 points), anchored
at 600 points = 50:1 odds; `factor = 20 / ln 2`, `offset = 600 - factor * ln 50`. The
intercept and offset are spread across characteristics to yield per-bin points.

**Points table:** [INSERT scorecard points table]

## 12. Assumptions & Limitations

- **No out-of-time validation.** The dataset lacks origination dates; only out-of-sample
  (train/test) validation is performed. A time-stamped panel is required for proper OOT
  testing and is identified as future work.
- **Single development dataset**, no external benchmark population.
- **No reject inference** — the model is fitted on accepted/observed loans only, so it may
  not represent the through-the-door population.
- **Target definition** is taken as the supplied flag rather than a regulatory DPD
  definition.

A thorough limitations statement is intentional: it bounds the model's valid use and
flags where independent validation should focus.

## 13. Ongoing Monitoring Plan

Proposed post-deployment monitoring:
- **Population Stability Index (PSI)** on the score distribution to detect population drift.
- **Characteristic Stability Index (CSI)** per feature to localise drift to specific drivers.
- **Periodic discrimination re-checks** (AUC/Gini/KS) on incoming performance data.
- **Calibration re-checks** with defined recalibration / redevelopment triggers.

## 14. Implementation & Reproducibility

The model is locked as a self-contained, reproducible package: the fitted binning object,
the fitted logistic regression model, the points table, and the development metrics, all
serialised and logged to **MLflow** tagged `champion`. This is the package handed to the
independent model-validation function and is sufficient to reproduce a score on new data.

## 15. Governance

| Item | Detail |
|------|--------|
| Model version | 1.0 |
| Developer | Daniel Cherkasskiy Saini |
| Development date | 14-06-2026 |
| Review / validation status | Pending independent validation |
| Code & artifact repository | https://github.com/danielsaini/retail-pd-scorecard-kaggle |

## Appendices

- B. WoE / binning tables
- C. Full coefficient output
- D. Decile calibration table
- E. Points (scorecard) table

## References

- Siddiqi, *Intelligent Credit Scoring*
- Baesens et al., *Credit Risk Analytics*
- Hosmer & Lemeshow, *Applied Logistic Regression*
- Federal Reserve / OCC, *SR 11-7: Guidance on Model Risk Management*
