# Hydro-informer

## Overview
This repository contains the data, code, and models for the Hydro-Informer project, a deep learning framework for predicting water levels and flood risks.

## Directory Structure
- `data/`: Contains the datasets used for training and testing.
- `src/`: Contains the source code for data preprocessing, model training, and evaluation.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis and model development [To be added].
- `models/`: Contains trained model files [To be added].

### Model
This model leverages Multi-Head Attention mechanisms along with Conv1D and LSTM layers to enhance the predictive performance by focusing on relevant features in the sequence data.

![Model 2 Architecture](figures/figure1.PNG)

### Predictions with Confidence Intervals
This plot shows the predicted water levels with confidence intervals, indicating the range within which the true water levels are expected to lie.

![Predictions with Confidence Intervals](figures/predictions_with_confidence_intervals.png)

### Peak Analysis
Comparative analysis of actual vs. predicted water levels for the most significant peaks observed in the testing data.

#### Peak Analysis Case 1
![Peak Analysis Case 1](figures/actual_vs_predicted_peak_comparison_case1__.png)

#### Peak Analysis Case 2
![Peak Analysis Case 2](figures/actual_vs_predicted_peak_comparison_case2__.png)

#### Peak Analysis Case 3
![Peak Analysis Case 3](figures/actual_vs_predicted_peak_comparison_case3___.png)

## Data
The dataset utilized for this study was obtained from the Slovak Hydrometeorological Institute, encompassing a comprehensive range of hourly measurements, including water levels, discharge rates, and precipitation levels.

## Installation
To install the required dependencies, run:
```sh
pip install -r requirements.txt

