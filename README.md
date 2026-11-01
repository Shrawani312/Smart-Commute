# Smart Commute ML

**Machine Learning Based Travel Time Prediction and Transport Recommendation System**

Author: Shrawani Thorat  
Programme: TE ENTC, PVGCOET&M, Pune

## Overview
This project extends an earlier Java/OOP commute-matching idea with a supervised machine-learning pipeline. It predicts commute time from trip distance, desired arrival time, traffic, weather, day and transport mode. It then compares Bike, Car, Bus and Shared Auto using predicted time and an estimated cost score.

## Pipeline
1. Generate a reproducible synthetic Pune commute dataset (608 rows before cleaning) or load `commute_dataset.csv` if supplied.
2. Inspect missing values, duplicates and invalid travel-time records.
3. Clean the data and engineer peak-hour, weekend, rain and traffic features.
4. Split the data into training and test sets.
5. Train Linear Regression and Random Forest pipelines with preprocessing.
6. Evaluate with MAE, RMSE, R² and 5-fold cross-validation.
7. Tune Random Forest and produce feature-importance and diagnostic plots.
8. Rank transport modes for a sample commute.

## Important data note
The included workflow can generate **synthetic data** using documented rules. The reported metrics are therefore demonstrations of the pipeline, not evidence of real-world predictive accuracy. Do not describe the dataset as collected survey data. Replace it with genuine, consented commute observations before making real-world claims.

## Run locally
```bash
python -m pip install -r requirements.txt
python smart_commute_ml.py
```

The script creates `commute_dataset.csv` (when no input CSV exists) and an `outputs/` directory containing plots, a demo recommendation and `results.json`.

## Example
Katraj → PVG College, 6.5 km, 9:00 AM, high traffic and rain. The script predicts travel time for each transport mode and ranks the options using a weighted time/cost score.

## Limitations and next steps
- Synthetic data may make the evaluation look more optimistic than real conditions.
- No live traffic, GPS, road-network routing or real fare feed is connected.
- Expand geographical coverage and collect a larger real dataset.
- Add uncertainty estimates and a deployed interface only after validating on real observations.
