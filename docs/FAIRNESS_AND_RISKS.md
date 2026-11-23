# Fairness, explainability, privacy and risk assessment (spec §3.13)

All numbers come from `python -m src.monitoring.fairness` (output `reports/fairness.json`), run on 2026-11-23. Both registered models were scored on the **same 1,409 held-out test customers** that are used for promotion. Confidence intervals come from a bootstrap with 1,000 resamples of the test set.

## 1. Sensitive attributes

| Attribute | Used as a model input? | Why we check it |
|---|---|---|
| `gender` | **No**: dropped in preprocessing (`configs/pipeline_config.yaml` → `drop_columns`) | Bias can still enter through correlated columns (proxies) |
| `SeniorCitizen` | **Yes**: it is one of the 18 inputs | Age is a protected characteristic in many jurisdictions |
| `customerID` | No: identifier only | Not personal data in this synthetic sample, but never used for prediction |

## 2. Group fairness results

Metrics per group: **recall** (share of real churners caught), **false positive rate (FPR)** (share of loyal customers wrongly flagged), **precision**, and **selection rate** (share of customers flagged).

### Production model: v2 Random Forest

| Group | n | Actual churn | Selection rate | Recall | FPR | Precision |
|---|---|---|---|---|---|---|
| Female | 687 | 28.1% | 38.0% | 74.6% | 23.7% | 55.2% |
| Male | 722 | 25.1% | 39.3% | 80.1% | 25.7% | 51.1% |
| Non-senior | 1,187 | 23.3% | 34.0% | 73.6% | 22.0% | 50.4% |
| **Senior** | 222 | **44.1%** | **64.0%** | **87.8%** | **45.2%** | 60.6% |

Selection rate = share of the group flagged as likely to churn.

### Is each gap real or noise? (bootstrap 95% CI of the difference)

| Model | Gap | Difference | 95% CI | Verdict |
|---|---|---|---|---|
| v2 | Recall, female − male | −5.5 pts | [−14.3, +3.2] | not significant |
| v2 | FPR, female − male | −2.0 pts | [−7.4, +3.0] | not significant |
| v2 | Recall, senior − non-senior | +14.2 pts | [+5.6, +22.2] | **significant** |
| v2 | **FPR, senior − non-senior** | **+23.2 pts** | **[+13.6, +32.4]** | **significant** |
| v1 | Recall, female − male | −1.3 pts | [−9.4, +6.6] | not significant |
| v1 | FPR, senior − non-senior | +26.0 pts | [+16.6, +34.7] | **significant** |

### Interpretation
- **Gender:** no evidence of a difference. All gender gaps are small, and their confidence intervals include 0, for both models. v2's 5.5-point recall gap is within noise for this sample size (about 190 churners per gender), so it should be **re-checked on more data** rather than treated as proven.
- **Senior citizens:** a clear and statistically significant difference. Seniors really do churn more (44% vs 23%), so a higher selection rate is partly justified by the data. But the model also **wrongly flags 45% of loyal seniors** against 22% of loyal non-seniors. This **violates equalized odds** (equal error rates across groups).
- **v1 vs v2:** v2 (the production model) slightly *reduces* the senior FPR gap (26.0 → 23.2 pts) and the selection-rate gap. On fairness, the promotion did not make things worse.

### Possible harm and mitigations
Whether a false "will churn" flag harms someone depends on the **business action**. If a flag triggers a **discount or a support call**, over-flagging seniors mostly costs the company money and can even benefit them. If it triggered something **negative** (less favourable terms, aggressive sales pressure, or excluding them from offers meant for "loyal" customers), the higher FPR would be a real harm to an age group.

Mitigations:
1. **Policy:** use the score only for *positive* retention actions, and document that in the model card / REPORT.
2. **Monitor fairness over time:** re-run `src.monitoring.fairness` for every candidate and check group FPR and recall gaps before promotion. This could become an additional promotion criterion.
3. **Technical options, if needed:** a group-aware decision threshold, re-weighting during training, or testing a model without `SeniorCitizen` and comparing accuracy against fairness. The last would still need a proxy check, because age correlates with other columns.
4. **Human review** of retention actions for flagged customers.

## 3. Explainability (permutation importance)

Drop in F1 on the test set when one input column is shuffled; a bigger drop means more important:

| Rank | v1 Logistic Regression | v2 Random Forest (production) |
|---|---|---|
| 1 | tenure (0.152) | **Contract** (0.106) |
| 2 | MonthlyCharges (0.031) | InternetService (0.037) |
| 3 | Contract (0.030) | tenure (0.033) |
| 4 | InternetService (0.027) | TotalCharges (0.028) |
| 5 | OnlineSecurity (0.004) | TechSupport (0.006) |

Both models rely on **contract type, tenure, internet service and charges**, which matches business intuition: new customers on month-to-month fiber contracts with high bills churn most. `SeniorCitizen` is **not** among the top drivers, so most of the senior gap comes from seniors' different contracts, services and charges rather than from the age flag itself. Logistic Regression is easier to explain to non-experts (one coefficient per feature), which is one reason it remains a reasonable fallback (rollback target v1).

## 4. Privacy, logging and data retention

- **No request bodies in logs.** The API logs only counts, model version and latency, plus gunicorn access lines. CI fails if a request value ever appears in the container logs (`ci.yml` smoke test).
- **Client IP addresses** appear in access logs (CloudWatch). They are personal data, so the log group keeps them **7 days**, and local Prometheus keeps metrics 7 days.
- **Secrets:** DagsHub token in the git-ignored `.env` locally, a Kubernetes Secret on kind, **AWS Secrets Manager** on AWS, and GitHub Secrets in CI. No AWS keys exist anywhere: GitHub deploys through **OIDC**.
- **Access control:** the AWS endpoint is reachable only from one allowed IP (security group `/32`). IAM roles are least-privilege (`deploy/aws/iam/`).
- **Input validation:** strict schema, 64 KB body limit, at most 100 instances per request, errors never echo input values.
- **Dataset licence:** see `DATASET_LICENSE.md`. It is IBM sample data used for non-commercial teaching; the permitted-use statement is our interpretation.

## 5. Risk assessment (project-specific)

| # | Risk | Likelihood / impact | Mitigation in this project | Residual risk |
|---|---|---|---|---|
| 1 | **Leaked credentials** (DagsHub token, AWS access) | Medium / High | Tokens only in git-ignored files, GitHub Secrets and Secrets Manager; OIDC instead of AWS keys; trust policy limited to this repo's `production` environment; public repo checked for secrets before each commit | A teammate's laptop `.env`; rotate the token after the course |
| 2 | **Age bias through the model** (seniors' FPR 45% vs 22%) | High (measured) / Medium–High depending on use | Measured with confidence intervals (§2); score restricted to positive retention actions; fairness check before promotion | Gap remains in the current model |
| 3 | **Silent data drift lowers accuracy** | Medium / High | KS / chi-square drift check with a 25% dataset-drift trigger (`src/monitoring/monitor.py`, exit code 2 → retraining); Grafana panels for prediction distribution and validation failures | Labels arrive late, so accuracy loss is detected indirectly |
| 4 | **A worse model is promoted, or bad code is deployed** | Medium / High | CI quality gates and branch protection; registry gate (`register.py` refuses failed or dirty runs); rule-based manual promotion (`promote.py`); GitHub environment approval for each deploy; ECS circuit-breaker rollback; one-command model rollback | Human approval can be wrong; v1 stays available for rollback |
| 5 | **Cloud cost overrun** | Low–Medium / Medium | $100 budget with 85% and 100% alerts; smallest Fargate size; service scaled to 0 between demos; ECR keeps 5 images; full cleanup runbook | Forgetting cleanup after the final demo |
| 6 | **Dependency vulnerabilities or supply-chain issues** | Medium / Medium | pip-audit, bandit, Trivy (CI) and ECR scan-on-push; pinned versions; one known dev-only advisory documented (diskcache via DVC) | New CVEs between scans |
| 7 | **Abuse of the public API** (floods, malformed input) | Low (IP-restricted) / Medium | Security group allows one IP; input validation and size limits; Prometheus alerts on errors, latency and validation failures | No rate limiting or authentication on the local demo endpoints |

## 6. Limitations

- The dataset is a small, older sample of fictional customers, so fairness numbers may not hold for a real customer base.
- Only two attributes were checked; intersectional groups (e.g. senior women) are too small in the test set for reliable estimates.
- Drift is **simulated** for the demo (see `configs/monitoring.yaml`); real labels and real traffic were not available.
