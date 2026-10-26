# Wearable and symptom modeling plan

## Working approach

- Explore features in `notebooks/03_wearable_feature_exploration.ipynb`.
- Build wearable models in `notebooks/04_wearables_modelling.ipynb`; use a separate symptom notebook later.
- Write no more than one notebook cell at a time, then pause for review and execution.
- Start with wearable feature exploration, without symptom predictors.
- Keep sensitive findings in ignored private notes and clear sensitive notebook outputs before committing.

## Existing implementation

`src/wearables.py` currently:

- Loads resting heart rate, skin temperature, HRV (RMSSD), and breathing rate.
- Aggregates readings into one row per participant-day.
- Inserts missing calendar days between each participant's first and last reading within each study interval.
- Creates eight features: each signal's deviation from its preceding 14-day median, and that deviation divided by its preceding standard deviation.
- Requires at least eight previous readings and excludes the current day from the reference window.
- Keeps reference windows separate across participants and study intervals.
- Joins features onto the daily prediction frame.

All 11 tests in `tests/test_wearables.py` passed during the initial review. They cover rolling calculations, gaps, participant and interval separation, and feature joins. Raw-file cleaning, aggregation, and nighttime alignment are not yet covered by those tests.

Existing assumptions to review rather than accept uncritically:

- Minimum daily resting heart rate as the daily summary.
- Excluding RMSSD readings at or above 300 ms.
- Moving HRV readings at or after 18:00 to the following day.
- Averaging multiple temperature readings ending on the same day.
- The 14-day reference window and eight-reading minimum.

## Stage 1: Wearable exploration

### 1. Define prediction task and timing

- Retain days until the next period as the target for comparison with the calendar baselines.
- Decide when predictions are made and which readings are available by that time.
- Keep the distinction between reported-flow-derived labels in Interval 1 and supplied-phase-derived labels in Interval 2 explicit.
- Define participant-held-out splits before making data-driven feature or modeling decisions.

### 2. Explore coverage

- Inspect available signals and source-table structure.
- Summarize coverage and missingness overall, by participant, and by study interval.
- Examine missing-day runs and the overlap of available signals.
- Distinguish raw-signal missingness from missing features caused by insufficient prior history.

### 3. Explore signal values and preprocessing

- Inspect distributions, outliers, and individual trajectories.
- Compare collection patterns between 2022 and 2024.
- Review aggregation and nighttime alignment against source definitions.
- Reassess filtering thresholds and add tests for accepted cleaning and alignment rules.

### 4. Explore within-person features

- Examine signal changes relative to each participant's own history.
- Inspect trailing deviations and standardized deviations, their correlations, and their availability.
- Explore relationships with cycle day using training participants for feature-selection decisions.
- Treat plots aligned using future period dates as retrospective exploration, not deployable predictors.

## Stage 2: Literature review and wearable model

### 5. Review research published in 2021–2026

Review primary studies using wearables or longitudinal tracking records to predict menstrual-cycle events. Record:

- Citation and publication year.
- Task: next menstruation prediction, ovulation prediction/detection, or phase classification.
- Cohort size, sensors, inputs, features, and required personal history.
- Model family and how predictions are updated over time.
- Prediction timing and whether the approach is prospective or retrospective.
- Participant/time splits, preprocessing, and leakage safeguards.
- Evaluation metrics, baselines, performance, and limitations.

Do not treat phase classification or retrospective ovulation detection as equivalent to forecasting the next period. Use the review to select a small set of models appropriate for this dataset's size; no model family is selected yet.

### 6. Build and evaluate wearable models

- Compare calendar-only with calendar-plus-wearable models.
- Use identical held-out participants and matched evaluation days for direct comparisons.
- Fit imputation, scaling, feature selection, and model tuning using training data only; keep tuning participant-grouped.
- Use only information available at prediction time, including for personal baselines and trailing features.
- Specify behavior when wearable data or prior history are missing; report prediction coverage alongside error.
- Compare performance over repeated participant splits, not just one favorable split.
- Report new versus returning participants, errors by cycle day, and interval-specific performance.
- Keep label-source differences explicit when interpreting pooled results.

## Stage 3: Symptom diary and combined model

### 7. Repeat exploration for symptoms

- Create a separate notebook after the wearable analysis.
- Examine symptom definitions, response categories, reporting timing, coverage, and participant/interval missingness.
- Distinguish an absent symptom from a missing diary entry.
- Identify which entries are available at prediction time and avoid fields that reveal the outcome retrospectively.
- Build and test symptom features before combining them with wearables.

### 8. Compare model input sets

Evaluate on matched held-out participants and days:

1. Calendar only.
2. Calendar plus wearables.
3. Calendar plus symptoms.
4. Calendar plus wearables and symptoms.

Report both performance and coverage so improvements are not confused with selecting an easier subset of participants or days.

## First notebook cell

Begin with one markdown cell defining the wearable notebook's scope and prediction task. Pause for review before adding the next cell.
