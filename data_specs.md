# In-depth overview and noteworthy details of mcPHASES dataset


## Cohort details
- Interval 1 from Jan 2022 - April 2022. Interval 2 from July 2024 - Oct 2024.
- Interval 1 cohort was 42 participants who recorder Fitbit data, CGM data, hormone levels via daily urinary panel, and self-reported diary of activity, sleep, stress, and bleeding.
- Interval 2 was 20 returning participants with only the hormone panel and fitbit tracker.
- Nearly 1/3 of Interval 1 study participants identified as having irregular periods (Table 1). *This may be used to further classify groups.*

## Data-processing details
- Dates will be represented as relative "day-in-study" values, with weekend/weekday data available.
- Missing data is not excluded
- There may have been updates to the fitbit or hormone panel between Interval 1 and 2 that obscure the algorithmically-derived measurements such as sleep duration.

## Dataset details
- Interval 1 and 2 are combined in the same dataset. Participant data, if in Interval 2 cohort, restarts at **day 838** (the study abstract says 905). Verified in `hormones_and_selfreport.csv`: Interval 2 `day_in_study` runs 838-1004.
- `study_interval` is coded as `2022` / `2024`, not 1 / 2.
- **Table 1**: demographics; **Table 2**: Index of data tables in the dataset with descriptions.
- Note for project consideration: missing / absent data may not be completely random. For fitbit data, participants were encouraged to take off tracker during showers, to encourage charging. Note that the time of absent data may not be random and processing should consider this fact.

## Bleeding and dates

- Interval 2 restarts at **day 838** (the study abstract says 905). Interval 2 `day_in_study` runs 838-1004.
- `study_interval` is coded `2022` / `2024`, not 1 / 2.
- There are no calendar dates. Every timestamp column in all 26 files is time of day only. The time axis is `day_in_study` plus the `is_weekend` flag.
- `day_in_study` increments at midnight, so a night's sleep that starts before midnight is split across two day indices. Group sleep-derived data by sleep session, not by day.
- There is no bleeding start field. Bleeding is recorded in `hormones_and_selfreport.flow_volume`, a daily self-report: `Not at all`, `Spotting / Very Light`, `Light`, `Somewhat Light`, `Moderate`, `Somewhat Heavy`, `Heavy`, `Very Heavy`. Null on 13.76% of Interval 1 days.
- Period start is derived: the first day of a run of bleeding, excluding `Not at all` and `Spotting / Very Light`.
- Interval 2 has no `flow_volume` at all. Its labels are taken from `phase == 'Menstrual'`, which is complete for all 1,961 rows. This is a supplied label rather than a self-report, and pooling the two intervals mixes the two label types.
- The gap between periods is set to more than 3 days, so a couple of unlogged days mid-period do not read as a new period. The usable cycle count moves very little across thresholds of 1 to 5.
- Cycle lengths are kept between 15 and 60 days rather than the clinical 21-35. About a third of this cohort reported irregular periods, and the tighter range drops 12 cycles and 7 of the participants who have enough history for a personalised model.
- `pdg` is absent from Interval 1 and `lh` and `estrogen` are around 94% present, so `lh` is the hormone signal used.

## Progress notes

- Interval 2 contains no bleeding reports and Interval 1 contains no `pdg`. Checked against all 23 tables and the dataset README.
- Pooling both intervals gives 119 cycles from 39 participants. Interval 1 alone gives 76 from 38.
