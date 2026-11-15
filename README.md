# Court Referral Decision Model for Unsecured Overdue Loans

Which overdue clients are worth taking to court? Legal proceedings cost money, most clients never repay enough to cover that cost, and the capacity of the legal department is limited. This project builds a model that estimates the expected recovery for each client, compares it with the court cost and recommends a referral only when the expected net result is positive.

The whole project is in one notebook: [`court_referral_model.ipynb`](court_referral_model.ipynb).

## Problem

The referral decision is made per client, and all of a client's unsecured loans go to court together. The court cost is a fixed 500 GEL per case plus 10% of the outstanding debt. Referring everyone loses money: on the in-scope population only about 16% of cases are profitable, and the "refer all" strategy ends at roughly -11 million GEL, while a perfect selection would reach about +10.8 million GEL. The value is in choosing the right cases, not in predicting recovery for its own sake.

## Data

Two tables covering 29,763 client-periods (about 22,000 clients), 2023/01 to 2026/02:

- loan level monthly recoveries for up to 24 months after the start of the court process;
- 50 anonymized client features (transaction activity, account balances, payment history on other loans, digital channels, credit bureau, declared income, recent credit applications).

The data is confidential and is not included in this repository.

## What made this harder than a standard regression

- **Partial outcomes.** More than half of the cases do not have the full 24 months yet. The 24 month result is extrapolated with a recovery curve built from fully observed cohorts, and the method is backtested (MAE of 0.86 percentage points at 12 months of observation).
- **Selection bias.** The data only contains clients that the expert rules already sent to court. The notebook is explicit about what can and cannot be concluded from that.
- **Repeat clients.** About a third of the clients appear in more than one period, so every split is done by time and by client.
- **Leakage.** Three features are recorded at or after the start of enforcement and are removed. One feature is constant, one is an exact duplicate, one drifts strongly over time (PSI 0.33).
- **Zero inflated target.** The median recovery rate is 0, so a direct regression is compared against a hurdle model (probability of any payment times expected rate given payment).

## Approach

1. Aggregate loans to the client-period level and keep only fully unsecured cases (23,171).
2. Target: 24 month recovery rate, capped at 1.0.
3. Evaluation in time ordered layers: Layer A (train and test both fully observed), Layer A', Layer B (most recent cohorts, extrapolated labels) and Layer C (the same recent cohorts scored only on actually observed 6 month recovery).
4. Random Forest, XGBoost and LightGBM, each as a direct and a hurdle model. Hyperparameters are selected by realised net GEL on expanding folds, not by RMSE or AUC.
5. Decision rule: `expected_net = predicted_rate * total_outstanding - (500 + 10% * total_outstanding)`, refer when it is positive. Cases below the 556 GEL break-even are never referred.

## Results

LightGBM direct was selected. XGBoost hurdle was statistically tied on net result (paired bootstrap CI includes zero, decisions agree in about 97% of cases), so the simpler single model won.

| | Layer A | Layer B |
| --- | ---: | ---: |
| Clients referred to court | 15.3% | 13.0% |
| Net result with the model | about 672k GEL | about 604k GEL |
| Net result of "refer all" | about -1.55M GEL | about -2.81M GEL |
| Improvement over "refer all" | 2.22M GEL | 3.41M GEL |
| Best simple rule (outstanding + bureau score) | about 200k GEL | about 51k GEL |

On the most recent cohorts the model keeps 53% of the recovery while paying 18% of the court cost. Annualized over the last 12 cohorts this is an uplift of roughly 5.8 to 6.5 million GEL against referring everyone.

The notebook also covers sensitivity analysis (decision margin, per loan cost, training on fully observed data only, number of trees), interpretation by segment, calibration by decile, a randomized pilot design with power calculations, and a monitoring plan.

## Limitations

- The model uses only about 29% to 40% of the oracle potential, and 57% of the referred cases are still loss-making on their own.
- All results are relative to "refer all" on a population the expert already approved. Nothing here says how the model compares with the expert on rejected clients.
- Labels of the most recent cohorts are partly extrapolated. Final validation needs a prospective pilot (section 11).

## Reproducibility

The data is confidential and is not part of this repository, so the notebook cannot be re-run from here. All outputs are saved in the notebook, so the full analysis, tables and charts can be read directly on GitHub. `requirements.txt` lists the libraries that were used (Python 3.12).

## Notebook outline

1. Data overview
2. Building the target variable
3. Feature analysis
4. Analysis of the dependent variable
5. Model frame and evaluation design
6. Modelling
7. Sensitivity analysis
8. Interpretation of the results
9. Decision rule and backtest
10. Potential business benefit
11. Pilot campaign plan
12. Model monitoring plan
