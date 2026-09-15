# Step 2, calendar-only baselines

119 cycles from 39 participants, 20 random splits, gap_days=3.

| model        |   n_splits |   mean_MAE |    sd |   best |   worst |
|:-------------|-----------:|-----------:|------:|-------:|--------:|
| calendar-day |         20 |      3.659 | 0.737 |  2.388 |   5.699 |
| population   |         20 |      3.691 | 0.788 |  2.409 |   5.787 |
| personalised |         20 |      3.927 | 0.954 |  2.47  |   6.399 |

## New against returning participants

| regime                   |   n_participants |   n_days |   personal_MAE |   population_MAE |
|:-------------------------|-----------------:|---------:|---------------:|-----------------:|
| new participant, cycle 0 |               12 |      614 |          5.956 |            5.956 |
| returning, cycle 1+      |               33 |     1815 |          3.473 |            3.015 |
