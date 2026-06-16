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
#### person_income

| Bin | Count | Count (%) | WoE | IV | JS |
|---|---|---|---|---|---|
| (-inf, 22890.00) | 1308 | 0.0502 | -1.8283 | 0.2240 | 0.0247 |
| [22890.00, 34990.00) | 3706 | 0.1422 | -0.8151 | 0.1145 | 0.0139 |
| [34990.00, 39937.50) | 1847 | 0.0709 | -0.2689 | 0.0055 | 0.0007 |
| [39937.50, 49994.00) | 3809 | 0.1461 | 0.0310 | 0.0001 | 0.0000 |
| [49994.00, 59736.00) | 3537 | 0.1357 | 0.0912 | 0.0011 | 0.0001 |
| [59736.00, 79993.50) | 5412 | 0.2076 | 0.4249 | 0.0330 | 0.0041 |
| [79993.50, 89325.00) | 1581 | 0.0607 | 0.9075 | 0.0377 | 0.0046 |
| [89325.00, inf) | 4864 | 0.1866 | 1.1223 | 0.1656 | 0.0197 |
| Special | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Totals** | 26064 | 1.0000 | — | 0.5817 | 0.0678 |

#### person_home_ownership

| Bin | Count | Count (%) | WoE | IV | JS |
|---|---|---|---|---|---|
| [OWN] | 2064 | 0.0792 | 1.2699 | 0.0858 | 0.0101 |
| [MORTGAGE] | 10765 | 0.4130 | 0.6674 | 0.1502 | 0.0184 |
| [OTHER, RENT] | 13235 | 0.5078 | -0.5062 | 0.1481 | 0.0183 |
| Special | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Totals** | 26064 | 1.0000 | — | 0.3841 | 0.0468 |

#### person_emp_length

| Bin | Count | Count (%) | WoE | IV | JS |
|---|---|---|---|---|---|
| (-inf, 0.50) | 3274 | 0.1256 | -0.3339 | 0.0153 | 0.0019 |
| [0.50, 1.50) | 2329 | 0.0894 | -0.2974 | 0.0086 | 0.0011 |
| [1.50, 2.50) | 3088 | 0.1185 | -0.2340 | 0.0069 | 0.0009 |
| [2.50, 4.50) | 5078 | 0.1948 | 0.0573 | 0.0006 | 0.0001 |
| [4.50, 5.50) | 2411 | 0.0925 | 0.2065 | 0.0037 | 0.0005 |
| [5.50, 6.50) | 2120 | 0.0813 | 0.2070 | 0.0033 | 0.0004 |
| [6.50, 7.50) | 1750 | 0.0671 | 0.2092 | 0.0028 | 0.0003 |
| [7.50, 11.50) | 3555 | 0.1364 | 0.2940 | 0.0108 | 0.0013 |
| [11.50, inf) | 1737 | 0.0666 | 0.4293 | 0.0108 | 0.0013 |
| Special | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 722 | 0.0277 | -0.5033 | 0.0080 | 0.0010 |
| **Totals** | 26064 | 1.0000 | — | 0.0708 | 0.0088 |

#### loan_intent

| Bin | Count | Count (%) | WoE | IV | JS |
|---|---|---|---|---|---|
| [VENTURE] | 4626 | 0.1775 | 0.4699 | 0.0341 | 0.0042 |
| [EDUCATION] | 5135 | 0.1970 | 0.2723 | 0.0135 | 0.0017 |
| [PERSONAL] | 4436 | 0.1702 | 0.1357 | 0.0030 | 0.0004 |
| [HOMEIMPROVEMENT] | 2852 | 0.1094 | -0.2204 | 0.0056 | 0.0007 |
| [MEDICAL] | 4854 | 0.1862 | -0.2802 | 0.0158 | 0.0020 |
| [DEBTCONSOLIDATION] | 4161 | 0.1596 | -0.3556 | 0.0222 | 0.0028 |
| Special | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Totals** | 26064 | 1.0000 | — | 0.0941 | 0.0117 |

#### loan_amnt

| Bin | Count | Count (%) | WoE | IV | JS |
|---|---|---|---|---|---|
| (-inf, 2025.00) | 1427 | 0.0547 | 0.1925 | 0.0019 | 0.0002 |
| [2025.00, 8237.50) | 12151 | 0.4662 | 0.1868 | 0.0154 | 0.0019 |
| [8237.50, 10587.50) | 3942 | 0.1512 | 0.1216 | 0.0022 | 0.0003 |
| [10587.50, 12837.50) | 2364 | 0.0907 | 0.0624 | 0.0003 | 0.0000 |
| [12837.50, 18087.50) | 3455 | 0.1326 | -0.2584 | 0.0095 | 0.0012 |
| [18087.50, 22150.00) | 1313 | 0.0504 | -0.5326 | 0.0164 | 0.0020 |
| [22150.00, inf) | 1412 | 0.0542 | -0.7459 | 0.0361 | 0.0044 |
| Special | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Totals** | 26064 | 1.0000 | — | 0.0818 | 0.0101 |

#### loan_int_rate

| Bin | Count | Count (%) | WoE | IV | JS |
|---|---|---|---|---|---|
| (-inf, 6.46) | 1697 | 0.0651 | 1.4837 | 0.0898 | 0.0103 |
| [6.46, 7.46) | 2010 | 0.0771 | 0.9487 | 0.0518 | 0.0062 |
| [7.46, 7.89) | 2035 | 0.0781 | 0.8647 | 0.0447 | 0.0054 |
| [7.89, 9.64) | 2562 | 0.0983 | 0.6042 | 0.0299 | 0.0037 |
| [9.64, 10.76) | 2948 | 0.1131 | 0.3192 | 0.0105 | 0.0013 |
| [10.76, 12.07) | 5881 | 0.2256 | 0.2325 | 0.0114 | 0.0014 |
| [12.07, 12.76) | 1628 | 0.0625 | 0.1276 | 0.0010 | 0.0001 |
| [12.76, 13.52) | 1922 | 0.0737 | -0.0143 | 0.0000 | 0.0000 |
| [13.52, 14.37) | 1564 | 0.0600 | -0.2548 | 0.0042 | 0.0005 |
| [14.37, 15.28) | 1418 | 0.0544 | -1.1295 | 0.0883 | 0.0105 |
| [15.28, inf) | 2399 | 0.0920 | -1.7143 | 0.3604 | 0.0402 |
| Special | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Totals** | 26064 | 1.0000 | — | 0.6920 | 0.0797 |

#### loan_percent_income

| Bin | Count | Count (%) | WoE | IV | JS |
|---|---|---|---|---|---|
| (-inf, 0.05) | 2775 | 0.1065 | 0.8911 | 0.0642 | 0.0078 |
| [0.05, 0.07) | 2178 | 0.0836 | 0.8217 | 0.0438 | 0.0053 |
| [0.07, 0.13) | 6820 | 0.2617 | 0.6661 | 0.0948 | 0.0116 |
| [0.13, 0.16) | 2024 | 0.0777 | 0.5403 | 0.0193 | 0.0024 |
| [0.16, 0.18) | 1881 | 0.0722 | 0.2896 | 0.0056 | 0.0007 |
| [0.18, 0.20) | 2382 | 0.0914 | 0.1944 | 0.0033 | 0.0004 |
| [0.20, 0.25) | 3018 | 0.1158 | 0.0966 | 0.0011 | 0.0001 |
| [0.25, 0.31) | 1925 | 0.0739 | -0.2019 | 0.0032 | 0.0004 |
| [0.31, 0.38) | 1671 | 0.0641 | -2.0543 | 0.3607 | 0.0385 |
| [0.38, inf) | 1390 | 0.0533 | -2.2540 | 0.3583 | 0.0372 |
| Special | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Totals** | 26064 | 1.0000 | — | 0.9542 | 0.1045 |

#### cb_person_default_on_file

| Bin | Count | Count (%) | Non-event | Event rate | WoE | IV | JS |
|---|---|---|---|---|---|---|---|
| [N] | 21493 | 0.8246 | 17536 | 0.1841 | 0.2123 | 0.0350 | 0.0044 |
| [Y] | 4571 | 0.1754 | 2842 | 0.3783 | -0.7795 | 0.1283 | 0.0156 |
| Special | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Missing | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Totals** | 26064 | 1.0000 | 20378 | 0.2182 | — | 0.1633 | 0.0200 |


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

Under maximum-likelihood estimation the logistic score equations force the mean
predicted PD to equal the observed default rate on the training sample, so aggregate
calibration is a structural property of the fit, not a tuned outcome. The holdout
result shows this property generalises:

| Partition | Mean predicted PD | Mean actual | Note |
|-----------|-------------------|-------------|------|
| Train | 0.218155 | 0.218155 | Equal by construction (MLE) |
| Test  | 0.2200 | 0.2182 | Aggregate calibration holds out-of-sample (+0.18pp, mildly conservative) |

At the decile level the model is well-calibrated in aggregate but shows a stable,
systematic deviation: modest over-prediction in the mid-to-upper bands (deciles 7–8,
+5.1pp and +4.9pp) and under-prediction at the lower-middle and extreme tail
(decile 3 −3.4pp, decile 10 −3.1pp). The same S-shaped pattern appears on the
training fold, indicating a functional-form characteristic of the logit link rather
than sampling noise. The tail direction (under-prediction of the highest-risk band)
is the non-conservative direction of error and is flagged for ongoing monitoring.

Calibration is reported here as validation evidence for the champion; no post-hoc
recalibration is applied, consistent with the by-construction property of the MLE
logistic fit.

## 11. Score Scaling (PDO)

Probabilities were scaled to an interpretable points system using the
Points-to-Double-the-Odds framework (Baesens,Bart):

`Score = offset + factor * ln(odds)`, with PDO = 20 (odds double every 20 points), anchored
at 600 points = 50:1 odds; `factor = 20 / ln 2`, `offset = 600 - factor * ln 50`. The
intercept and offset are spread across characteristics to yield per-bin points.

**Points table:** 
### Scorecard

**Scaling parameters**

| Parameter | Value |
|---|---|
| Intercept (α) | −1.3674 |
| Factor | 28.8539 |
| PDO (points to double the odds) | 20 |
| Number of features (N) | 8 |
| Offset / N (per-feature base) | 60.8902 |
| Neutral points per feature (WoE = 0) | 65.8221 |

**Points formula (per bin)**

```
Points = Offset/N − Factor × ( Intercept/N + Coef × WoE )
       = 60.8902 − 28.8539 × ( −0.170925 + Coef × WoE )
```

The intercept and offset are allocated equally across all 8 features, so every WoE = 0
bin (Special, and Missing where development had no missing records) scores 65.8221.

**Scorecard points table**

| Feature | Bin | WoE | Coef | Points |
|---|---|---|---|---|
| person_income | (-inf, 22890.00) | -1.8283 | -0.8260 | 22.2480 |
| person_income | [22890.00, 34990.00) | -0.8151 | -0.8260 | 46.3949 |
| person_income | [34990.00, 39937.50) | -0.2689 | -0.8260 | 59.4131 |
| person_income | [39937.50, 49994.00) | 0.0310 | -0.8260 | 66.5605 |
| person_income | [49994.00, 59736.00) | 0.0912 | -0.8260 | 67.9961 |
| person_income | [59736.00, 79993.50) | 0.4249 | -0.8260 | 75.9494 |
| person_income | [79993.50, 89325.00) | 0.9075 | -0.8260 | 87.4508 |
| person_income | [89325.00, inf) | 1.1223 | -0.8260 | 92.5715 |
| person_income | Special | 0.0000 | -0.8260 | 65.8221 |
| person_income | Missing | 0.0000 | -0.8260 | 65.8221 |
| person_home_ownership | [OWN] | 1.2699 | -0.8923 | 98.5156 |
| person_home_ownership | [MORTGAGE] | 0.6674 | -0.8923 | 83.0059 |
| person_home_ownership | [OTHER, RENT] | -0.5062 | -0.8923 | 52.7886 |
| person_home_ownership | Special | 0.0000 | -0.8923 | 65.8221 |
| person_home_ownership | Missing | 0.0000 | -0.8923 | 65.8221 |
| person_emp_length | (-inf, 0.50) | -0.3339 | -0.2956 | 62.9741 |
| person_emp_length | [0.50, 1.50) | -0.2974 | -0.2956 | 63.2857 |
| person_emp_length | [1.50, 2.50) | -0.2340 | -0.2956 | 63.8260 |
| person_emp_length | [2.50, 4.50) | 0.0573 | -0.2956 | 66.3104 |
| person_emp_length | [4.50, 5.50) | 0.2065 | -0.2956 | 67.5830 |
| person_emp_length | [5.50, 6.50) | 0.2070 | -0.2956 | 67.5876 |
| person_emp_length | [6.50, 7.50) | 0.2092 | -0.2956 | 67.6065 |
| person_emp_length | [7.50, 11.50) | 0.2940 | -0.2956 | 68.3295 |
| person_emp_length | [11.50, inf) | 0.4293 | -0.2956 | 69.4836 |
| person_emp_length | Special | 0.0000 | -0.2956 | 65.8221 |
| person_emp_length | Missing | -0.5033 | -0.2956 | 61.5300 |
| loan_intent | [VENTURE] | 0.4699 | -1.3910 | 84.6816 |
| loan_intent | [EDUCATION] | 0.2723 | -1.3910 | 76.7507 |
| loan_intent | [PERSONAL] | 0.1357 | -1.3910 | 71.2680 |
| loan_intent | [HOMEIMPROVEMENT] | -0.2204 | -1.3910 | 56.9764 |
| loan_intent | [MEDICAL] | -0.2802 | -1.3910 | 54.5771 |
| loan_intent | [DEBTCONSOLIDATION] | -0.3556 | -1.3910 | 51.5494 |
| loan_intent | Special | 0.0000 | -1.3910 | 65.8221 |
| loan_intent | Missing | 0.0000 | -1.3910 | 65.8221 |
| loan_amnt | (-inf, 2025.00) | 0.1925 | -0.3549 | 67.7929 |
| loan_amnt | [2025.00, 8237.50) | 0.1868 | -0.3549 | 67.7349 |
| loan_amnt | [8237.50, 10587.50) | 0.1216 | -0.3549 | 67.0673 |
| loan_amnt | [10587.50, 12837.50) | 0.0624 | -0.3549 | 66.4610 |
| loan_amnt | [12837.50, 18087.50) | -0.2584 | -0.3549 | 63.1760 |
| loan_amnt | [18087.50, 22150.00) | -0.5326 | -0.3549 | 60.3686 |
| loan_amnt | [22150.00, inf) | -0.7459 | -0.3549 | 58.1843 |
| loan_amnt | Special | 0.0000 | -0.3549 | 65.8221 |
| loan_amnt | Missing | 0.0000 | -0.3549 | 65.8221 |
| loan_int_rate | (-inf, 6.46) | 1.4837 | -1.1835 | 116.4871 |
| loan_int_rate | [6.46, 7.46) | 0.9487 | -1.1835 | 98.2192 |
| loan_int_rate | [7.46, 7.89) | 0.8647 | -1.1835 | 95.3504 |
| loan_int_rate | [7.89, 9.64) | 0.6042 | -1.1835 | 86.4531 |
| loan_int_rate | [9.64, 10.76) | 0.3192 | -1.1835 | 76.7225 |
| loan_int_rate | [10.76, 12.07) | 0.2325 | -1.1835 | 73.7622 |
| loan_int_rate | [12.07, 12.76) | 0.1276 | -1.1835 | 70.1793 |
| loan_int_rate | [12.76, 13.52) | -0.0143 | -1.1835 | 65.3339 |
| loan_int_rate | [13.52, 14.37) | -0.2548 | -1.1835 | 57.1212 |
| loan_int_rate | [14.37, 15.28) | -1.1295 | -1.1835 | 27.2518 |
| loan_int_rate | [15.28, inf) | -1.7143 | -1.1835 | 7.2811 |
| loan_int_rate | Special | 0.0000 | -1.1835 | 65.8221 |
| loan_int_rate | Missing | 0.0000 | -1.1835 | 65.8221 |
| loan_percent_income | (-inf, 0.05) | 0.8911 | -0.9795 | 91.0065 |
| loan_percent_income | [0.05, 0.07) | 0.8217 | -0.9795 | 89.0458 |
| loan_percent_income | [0.07, 0.13) | 0.6661 | -0.9795 | 84.6479 |
| loan_percent_income | [0.13, 0.16) | 0.5403 | -0.9795 | 81.0927 |
| loan_percent_income | [0.16, 0.18) | 0.2896 | -0.9795 | 74.0068 |
| loan_percent_income | [0.18, 0.20) | 0.1944 | -0.9795 | 71.3155 |
| loan_percent_income | [0.20, 0.25) | 0.0966 | -0.9795 | 68.5534 |
| loan_percent_income | [0.25, 0.31) | -0.2019 | -0.9795 | 60.1149 |
| loan_percent_income | [0.31, 0.38) | -2.0543 | -0.9795 | 7.7627 |
| loan_percent_income | [0.38, inf) | -2.2540 | -0.9795 | 2.1194 |
| loan_percent_income | Special | 0.0000 | -0.9795 | 65.8221 |
| loan_percent_income | Missing | 0.0000 | -0.9795 | 65.8221 |
| cb_person_default_on_file | [N] | 0.2123 | -0.1865 | 66.9647 |
| cb_person_default_on_file | [Y] | -0.7795 | -0.1865 | 61.6271 |
| cb_person_default_on_file | Special | 0.0000 | -0.1865 | 65.8221 |
| cb_person_default_on_file | Missing | 0.0000 | -0.1865 | 65.8221 |

Total score range: approximately 317 (all worst bins) to 688 (all best bins).
WoE convention: higher WoE = lower risk, so the negative coefficients are the
expected sign (higher WoE lowers the log-odds of default).

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


## References

- Siddiqi, *Intelligent Credit Scoring*
- Baesens et al., *Credit Risk Analytics*
- Hosmer & Lemeshow, *Applied Logistic Regression*
- Federal Reserve / OCC, *SR 11-7: Guidance on Model Risk Management*
