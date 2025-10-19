# NBA Player Performance Prediction - Model Evaluation Report

Generated: 2025-10-18 22:05:43

## Test Set Performance

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| PTS      | Lasso   | test    | 5.774   | 7.36141 | 0.376761 |        6.00775 |           3.89086 |        4385 |
| REB      | XGBoost | test    | 2.18541 | 2.83308 | 0.421004 |        2.2239  |           1.73067 |        4385 |
| AST      | XGBoost | test    | 1.76156 | 2.31352 | 0.436163 |        1.80885 |           2.61431 |        4385 |

## PTS - All Splits

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| PTS      | Lasso   | train   | 5.74581 | 7.34378 | 0.372278 |        5.96319 |           3.64531 |       15811 |
| PTS      | Lasso   | val     | 5.73934 | 7.25316 | 0.373527 |        5.94337 |           3.43292 |        3129 |
| PTS      | Lasso   | test    | 5.774   | 7.36141 | 0.376761 |        6.00775 |           3.89086 |        4385 |

## REB - All Splits

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| REB      | XGBoost | train   | 2.09334 | 2.70973 | 0.442941 |        2.17549 |           3.77588 |       15811 |
| REB      | XGBoost | val     | 2.10343 | 2.75066 | 0.436262 |        2.1396  |           1.69039 |        3129 |
| REB      | XGBoost | test    | 2.18541 | 2.83308 | 0.421004 |        2.2239  |           1.73067 |        4385 |

## AST - All Splits

| target   | model   | split   |     mae |    rmse |       r2 |   baseline_mae |   improvement_pct |   n_samples |
|:---------|:--------|:--------|--------:|--------:|---------:|---------------:|------------------:|------------:|
| AST      | XGBoost | train   | 1.70536 | 2.21798 | 0.468946 |        1.75626 |           2.89794 |       15811 |
| AST      | XGBoost | val     | 1.75588 | 2.28956 | 0.434969 |        1.78901 |           1.85188 |        3129 |
| AST      | XGBoost | test    | 1.76156 | 2.31352 | 0.436163 |        1.80885 |           2.61431 |        4385 |

