# Data Scale Experiment Results

**Date:** 2025-10-20 17:09:14

## Objective

Test how the amount of data (number of seasons and players) affects NBA player performance prediction models.

## Configurations Tested

| Experiment   | Name                       |   Players |   Games |   Train |   Test |   PTS_MAE |   PTS_Improve |   REB_MAE |   REB_Improve |   AST_MAE |   AST_Improve |
|:-------------|:---------------------------|----------:|--------:|--------:|-------:|----------:|--------------:|----------:|--------------:|----------:|--------------:|
| EXP_60P      | Top 60 Players             |        60 |   12676 |    8457 |   2496 |   6.02636 |       4.35321 |   2.28314 |       1.49168 |   1.87123 |       3.39638 |
| EXP_90P      | Top 90 Players             |        90 |   18562 |   12448 |   3580 |   5.88564 |       3.76966 |   2.19809 |       1.56409 |   1.77683 |       2.70626 |
| EXP_120P     | Top 120 Players (Baseline) |       120 |   23325 |   15811 |   4385 |   5.774   |       3.89086 |   2.18541 |       1.73067 |   1.76156 |       2.61431 |
| EXP_2S       | 2 Seasons Only             |       120 |   15740 |    8226 |   4385 |   5.77674 |       3.84533 |   2.19316 |       1.3823  |   1.77439 |       1.90496 |

## Key Findings

### Best Configurations by Target


**PTS:**
- Best: Top 120 Players (Baseline) (MAE: 5.774, Improvement: +3.9%)
- Players: 120, Games: 23,325

**REB:**
- Best: Top 120 Players (Baseline) (MAE: 2.185, Improvement: +1.7%)
- Players: 120, Games: 23,325

**AST:**
- Best: Top 120 Players (Baseline) (MAE: 1.762, Improvement: +2.6%)
- Players: 120, Games: 23,325

## Visualizations

- `experiment_comparison.png`: Side-by-side comparison of all configurations
- `data_scale_effect.png`: Performance vs data scale scatter plots

## Conclusion

[To be filled based on findings]
