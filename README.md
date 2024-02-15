# Project README

## Overview

This project explores the relationship between a country's GDP per capita and its citizens' life satisfaction. The goal is to create a linear model that predicts life satisfaction based on GDP per capita. The data used for this analysis is from the OECD Better Life Index and GDP per capita datasets.



## Introduction <a name="introduction"></a>

The question of whether money makes people happier is explored by analyzing data from various countries. The focus is on understanding the trend between life satisfaction and GDP per capita. The project uses a simple linear model to represent this relationship, where life satisfaction is modeled as a function of GDP per capita.

## Model Selection <a name="model-selection"></a>

The selected model is a simple linear model expressed by the equation:

``` 
life_satisfaction = θ₀ + θ₁ * GDP_per_capita
```


This linear model has two parameters, θ0 and θ1, which can be adjusted to fit the data. The model aims to capture the linear relationship between GDP per capita and life satisfaction.

## Linear Regression <a name="linear-regression"></a>

The Linear Regression algorithm is employed to find the optimal parameter values (θ0 and θ1) that minimize the distance between the model's predictions and the training examples. The algorithm is used to train the model on the provided data, resulting in parameter values that best fit the linear model to the data.

## Example Code <a name="example-code"></a>

To illustrate the process, the project includes Python code using the Scikit-Learn library. The code:

1. Loads the OECD Better Life Index and GDP per capita datasets.
2. Prepares the data for analysis.
3. Creates a scatterplot for visualization.
4. Trains a linear model using the Linear Regression algorithm.
5. Makes a prediction for a specific country (Cyprus) based on its GDP per capita.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new))  # outputs [[ 5.96242338]]

```

## Usage <a name="usage"></a>

To use this project for similar analyses, follow these steps:

Ensure you have the required dependencies installed (see Dependencies).
Download the project files and datasets.
Customize the code for your specific datasets or analysis requirements.
Run the code to visualize the data, train the linear model, and make predictions.

## Dependencies <a name="dependencies"></a>

<ul>
  <li>Python (>=3.6)</li>
  <li>NumPy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Scikit-Learn</li>
</ul>
