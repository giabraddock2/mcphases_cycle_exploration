# mcphases_cycle_exploration
## Predicting Period Start Dates with Wearable Data

Use mcPHASES dataset to predict cycle timing in 42 women. 

*cycle timing* is defined as the number of days until the participant's next reported period begins. The prediction will be made once at the end of the day, using data available on that day. The goal of the project is to refine precision and accuracy on wearable-accessible data from the input data, including, hear rate, HRV, sleep, activity, skin temperature, and past calendar history of cycles. We can use the additional symptomatic and hormonal data from the dataset to compare the wearable specific model.

What counts as a period? first day of bleeding, excluding spotting.

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

via PhysioNet

## Plan

### 1. Build rudimentary estimates

* Use person’s average cycle length / length of the person’s previous cycle

### 2. Add wearable data

* Resting heart rate
* Heart rate variability
* Skin temperature
* Sleep
* Activity
* Breathing rate

### 3. Compare different models

* Linear or logistic regression
* Random forest
* Gradient boosting
* Time-based model

### 4. Reduce the available data

* Calendar data only
* Basic wearable data
* Data commonly available from Oura
* All available Fitbit data
* Wearable data plus symptoms
* Wearable data plus hormones

### 5. Compare two use cases

* Predicting for a new person
* Predicting for someone after seeing their earlier cycles

### 6. Test synthetic data

Create realistic fake cycles and use them as extra training data.

The final test will always use real data from people the model has not seen.

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
notebooks/     01_data_exploration, 02_baselines
src/           data.py, baselines.py, evaluation.py
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

- Notebooks show aggregates and plots only, never participant rows.
- Notebook outputs are cleared before committing.

## Progress

- Step 1, rudimentary estimates: the population and personalised baselines in notebook 02.
- Step 2, three calendar-only models: population (one cycle length for everyone), personalised
  (each participant's own earlier cycles) and calendar-day (average days remaining at each cycle
  day, with no assumed length). Results in `outputs/results/baseline_results.md`.
- Step 3, wearable features: not started.
