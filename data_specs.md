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
- Interval 1 and 2 are combined in the same dataset. Participant data, if in Interval 2 cohort, restarts at day 905.
- **Table 1**: demographics; **Table 2**: Index of data tables in the dataset with descriptions.
- Note for project consideration: missing / absent data may not be completely random. For fitbit data, participants were encouraged to take off tracker during showers, to encourage charging. Note that the time of absent data may not be random and processing should consider this fact.
