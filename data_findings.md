# Data findings

Measured from the CSVs. Scripts are in `notebooks/01_data_exploration.ipynb`.
Interval 1 unless stated.

## Missing data

1. **Columns where 0 = NaN**: `resting_heart_rate` (should be > 0 always), 782 of 3,698 rows,
   29 of 42 participants. Real values run 47.6-88.1 bpm. `isna()` reports this column as 0% missing.
   `stress_score` marks the same thing as `status == 'NO_DATA'`, 755 of 5,941 rows.
2. **Rows absent**: days omitted where the wearable produced no signal. Not all zeros are gaps —
   `time_in_heart_rate_zones.in_default_zone_3` is 0 on 93.6% of days because most people spend no
   time in the peak zone.
3. **Columns with nulls**: listed at the bottom.

Expected total is 3,698 participant-days: 40 participants at 90 days, one at 60, one at 38.

| table | participants | participant-days | days missing |
|---|---:|---:|---:|
| **exercise** | 24 | 540 | **85.4%** |
| **stress_score** | 31 | 2,339 | **36.7%** |
| **respiratory_rate_summary** | 40 | 2,845 | **23.1%** |
| **heart_rate_variability_details** | 40 | 2,878 | **22.2%** |
| **computed_temperature** | 42 | 2,948 | **20.3%** |
| active_zone_minutes | 42 | 3,005 | 18.7% |
| altitude | 42 | 3,037 | 17.9% |
| glucose | 42 | 3,109 | 15.9% |
| sleep | 42 | 3,172 | 14.2% |
| wrist_temperature | 42 | 3,190 | 13.7% |
| sleep_score | 42 | 3,248 | 12.2% |
| heart_rate | 41 | 3,483 | 5.8% |
| time_in_heart_rate_zones | 42 | 3,489 | 5.7% |
| estimated_oxygen_variation | 42 | 3,496 | 5.5% |
| demographic_vo2_max | 42 | 3,531 | 4.5% |
| distance | 42 | 3,573 | 3.4% |
| steps | 42 | 3,573 | 3.4% |
| calories | 42 | 3,695 | 0.1% |
| active_minutes | 42 | 3,698 | 0% |
| hormones_and_selfreport | 42 | 3,698 | 0% |
| resting_heart_rate | 42 | 3,698 | 0% (see point 1) |

Columns with nulls:

| table | column | % null |
|---|---|---:|
| hormones_and_selfreport | `pdg` | 100 |
| | `flow_volume`, `flow_color` | 13.76 |
| | 11 symptom columns | 10.22 |
| | `estrogen`, `lh` | ~6 |
| height_and_weight | 2024 height / weight | 81 / 74 |
| | 2022 height / weight | 50 / 43 |
| exercise | `elevationgain` | 9.23 |
| computed_temperature | 4 baseline columns | 2.09 |
| subject-info | menstrual health literacy | 2.38 |

`flow_volume` determines the flow label.

## Table shape

Participant id is a column, not a column per participant. One wearable table looks like this:

```
id,study_interval,is_weekend,day_in_study,timestamp,temperature_diff_from_baseline
1,2022,False,3,00:00:00,0.0239130434782595
1,2022,False,3,00:01:00,-0.0210869565217421
1,2022,False,3,00:02:00,-0.0134782608695652
```

## Hormones

`lh` is the hormone signal used: 94.0% present in Interval 1, and it peaks around 14 days before
the next period. `estrogen` is similarly available but its highest reading lands in the second half
of the cycle in 55% of cycles, so it does not locate ovulation on its own. `pdg` is absent from
Interval 1.

Tested on 74 Interval 1 cycles, leave-one-cycle-out per participant:

| anchor | MAE (days) | within 2d |
|---|---:|---:|
| own mean cycle length | 4.08 | 45% |
| LH peak | 3.06 | 68% |
| estrogen peak | 5.05 | 45% |
