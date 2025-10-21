# NBA Player Performance Prediction - Model Evaluation Report

Generated: 2025-10-21 00:22:01

## Test Set Performance

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| PTS      | Lasso   | test    | 5.44795 | 6.99854 | 0.412717 |        5.65458 |           3.65421 |        7112 |
| REB      | XGBoost | test    | 2.13372 | 2.78195 | 0.426202 |        2.18566 |           2.3763  |        7112 |
| AST      | XGBoost | test    | 1.64188 | 2.16876 | 0.451618 |        1.67927 |           2.22668 |        7112 |

## PTS - All Splits

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| PTS      | Lasso   | train   | 5.37443 | 6.88519 | 0.442558 |        5.55759 |           3.29579 |       44600 |
| PTS      | Lasso   | val     | 5.42689 | 6.86545 | 0.411948 |        5.62208 |           3.47182 |        5100 |
| PTS      | Lasso   | test    | 5.44795 | 6.99854 | 0.412717 |        5.65458 |           3.65421 |        7112 |

## REB - All Splits

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| REB      | XGBoost | train   | 2.09475 | 2.71424 | 0.439767 |        2.15276 |           2.69468 |       44600 |
| REB      | XGBoost | val     | 2.07401 | 2.70072 | 0.436246 |        2.10933 |           1.67477 |        5100 |
| REB      | XGBoost | test    | 2.13372 | 2.78195 | 0.426202 |        2.18566 |           2.3763  |        7112 |

## AST - All Splits

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| AST      | XGBoost | train   | 1.57953 | 2.09119 | 0.502309 |        1.61532 |           2.21538 |       44600 |
| AST      | XGBoost | val     | 1.62385 | 2.13179 | 0.453389 |        1.66471 |           2.45428 |        5100 |
| AST      | XGBoost | test    | 1.64188 | 2.16876 | 0.451618 |        1.67927 |           2.22668 |        7112 |

