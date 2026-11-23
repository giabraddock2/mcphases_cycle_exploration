# mcphases_cycle_exploration

## Predicting Period Start Dates with Wearable Data

Explore cycle-timing prediction using the mcPHASES dataset of 42 participants.

*Cycle timing* is defined as the number of days until the participant's next recorded period begins. Models make daily predictions using past observations available at the prediction cutoff. The current wearable experiments use prior-day signals, with an additional lag for night-start-labelled HRV. The project investigates whether heart rate, HRV, breathing rate and skin temperature improve predictions beyond cycle history. Sleep, activity, symptoms and hormones are potential future extensions.

For reported flow, a period starts on the first day of bleeding, excluding spotting. Interval 2 instead uses supplied menstrual-phase labels as an onset proxy.

## Data

This project uses the [mcPHASES dataset](https://physionet.org/content/mcphases/1.0.0/).

It includes data from 42 people, including:

* Period dates
* Hormone levels
* Heart rate
* Heart rate variability
* Skin temperature
* Sleep
* Activity
* Stress
* Symptoms

## Overview of Notebooks

### 1. Data Exploration

- Investigated participant coverage, missingness, spotting, bleeding report coverage and cycle lengths across two study intervals.
- Identified a difference in target labels: Interval 1 provides self-reported flow, while Interval 2 lacks flow data and uses supplied menstrual-phase labels. Interval-specific experiments were therefore evaluated separately, alongside exploratory pooled comparisons.
- Constructed onset-to-onset cycles, excluded spotting from bleeding labels, examined gaps in bleeding reports and excluded cycles outside 15–60 days. The current baseline dataset contains 119 eligible cycles.
- Created one prediction row per day within each eligible cycle.

### 2. Baseline Models

- Implemented population-cycle-length, calendar-day and personalised-cycle baselines, with fallback behaviour when personal history is unavailable.
- Compared test (held-out) participants across 20 splits and inspected errors by participant, cycle position and history availability.
- Pooled-interval results for the calendar-day and population baseline models are near **3.66 and 3.69 days MAE**, respectively. Individual-participant differences were much larger than this gap.

**Finding:** simple history/calendar baselines were competitive; personalisation was not automatically beneficial with short histories. Lower error on later observed cycles does not by itself establish that personalisation caused the improvement.

### 3. Wearable feature exploration

- Investigated resting HR, HRV, skin temperature and breathing rate, including distributions, recording frequency, calendar gaps, coverage across cycles and near menstruation, and within-person correlations.
- Distinguished missing values from entirely absent days and invalid zeros (e.g. a breathing rate of 0 is treated as missing).
- Found unusual repetition of max HR and error values across participants in Interval 2. This raised a quality concern, not a confirmed dataset defect; prefer Interval 1 until clarification.
- Visualised signals and rolling features within participants.

### 4. Wearable Models and Results

Compared ridge regression, random forests and gradient boosting using participant-held-out tests, grouped inner tuning and training-fold imputation.

| Experiment | Main result / Finding |
|---|---|
| Interval 1 random forest: history alone versus history plus 3-day or 6-day wearable summaries (median, maximum, minimum, slope and SD). | History MAE **3.987**; 3-day wearable set **4.209**; 6-day set **4.277**. More wearable features did not mean better performance. |
| Add individual wearable features to the Interval 1 random forest. | Three-day maximum breathing rate reduced MAE to **3.880**, a **2.7%** improvement, with improvement in **16/20** splits. Three-day median also improved MAE. |
| Check breathing-rate benefits across cohorts and models. | Benefits did not consistently persist with pooled-interval random forests or Interval 2 gradient boosting. |
| Compare absolute-error and squared-error loss in Interval 2 random forests after observing late near-period predictions. | Absolute-error training improved history MAE from **3.119 to 3.048**, and final-three-day MAE from **2.991 to 2.485**. |
| Absolute-error gradient boosting on Interval 2. | History-only MAE was **3.015**. Adding wearable inputs did not consistently improve it. |

Additional experiments included:

- Tested random-forest minimum leaf sizes of 1, 3, 5, 10 and 30. Smaller leaves did not improve held-out performance in this comparison; near-period predictions remained late.
- Compared imputation methods and evaluated models on matching test rows with versus without observations. These checks did not establish missingness as the sole explanation for weak wearable performance.
- Compared rolling temperature baselines and physiologically motivated sustained-rise features. Sustained rise was more promising for some participants.
- Investigated whether benefits differed with cycle variability. In Interval 1, sustained-temperature-rise features reduced participant-weighted MAE from **6.039 to 5.905** in seven variable-cycle participants and **2.796 to 2.744** in 23 less-variable participants. The difference in benefit between groups was not statistically established. Here, variable cycles means an observed cycle-length range >7 days, not a clinical diagnosis; participants with only one cycle were unclassified.

These are exploratory comparisons on reused test splits, not confirmatory findings. The 20 splits overlap and are not independent trials. Feature selection needs fresh evaluation or nested validation before making a final performance claim. Daily and participant-weighted MAEs use different weighting; comparisons across cohorts are not controlled model comparisons.

## Main Output

* Average error in days
* Median error in days
* Percent of predictions within 1, 2, and 3 days
* Results for each person
* Results at different points in the cycle

---

## Repository layout

```
data/          local only, git-ignored
  raw/         unzipped mcPHASES CSVs
  processed/
notebooks/     01_data_exploration, 02_baselines,
               03_wearable_feature_exploration, 04_wearables_modelling
src/           data.py, baselines.py, evaluation.py, wearables.py
outputs/       figures/ and results/
tests/
```

## Setup

```bash
pip install -r requirements.txt
unzip mcphases-*.zip -j -d data/raw/
python -m pytest tests/ -q
```

## Data handling

The mcPHASES data is PhysioNet credentialed access.

- Raw data and private notes remain local and Git-ignored.
- Local notebook outputs may include participant-level diagnostics; outputs are cleared before committing.
- Clearing outputs does not remove findings written into Markdown or source code. Review those separately before publication.
