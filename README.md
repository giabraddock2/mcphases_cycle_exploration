# mcphases_cycle_exploration
## Predicting Period Start Dates with Wearable Data

Use mcPHASES dataset to predict cycle timing in 42 women. Use dataset to explore the strength of models with limited data and increasingly limited features.

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
