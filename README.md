# Predicting Song Release Year using Linear Regression with Spark and the Million Song Dataset

## Overview

This project focuses on building a linear regression model to predict the release year of songs using the Million Song Dataset. The goal is to develop a comprehensive supervised learning pipeline with PySpark, from data loading and parsing to model training and evaluation.

## Dataset

The dataset used in this project is a subset of the [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/), available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD). It contains audio features for songs, along with their release years.

## Contents

The project is divided into five main parts:

1. **Reading & Parsing the Original Dataset**
    - Loading and verifying the dataset.
    - Visualization of features and shifting labels.

2. **Development and Assessment of a Baseline Model**
    - Creating a baseline linear regression model.
    - Visualization comparing predicted vs. actual results.

3. **Training with Gradient Descent**
    - Training a linear regression model using gradient descent.
    - Evaluation and visualization of training errors.

4. **Training with MLlib and Hyperparameter Optimization**
    - Training the model with Spark MLlib.
    - Performing hyperparameter optimization using grid search.
    - Visualization of predictions and hyperparameter heat map.

5. **Adding Interactions Between Features**
    - Enhancing the model by including feature interactions.

## Visualizations

The notebook includes several visualizations to aid in understanding the data and the model's performance:

1. **Feature Visualization**: Initial exploration of the dataset's features.
2. **Label Shifting Visualization**: Analysis of label distribution.
3. **Predicted vs. Actual Results**: Comparison plot for the baseline model.
4. **Training Errors Examination**: Visualization of training errors during gradient descent.
5. **Best Model Predictions**: Predictions from the optimized model.
6. **Hyperparameter Heat Map**: Representation of the hyperparameter optimization results.

## Code Highlights

- **Data Loading**: Transforming raw data into an RDD format and verifying its contents.
- **Baseline Model**: Development and assessment of a simple linear regression model.
- **Gradient Descent**: Implementation and evaluation of a gradient descent training process.
- **MLlib Training**: Utilization of Spark MLlib for model training and hyperparameter tuning.
- **Feature Interactions**: Inclusion of interactions between features to enhance model performance.

## Usage

1. **Environment Setup**: Ensure PySpark and necessary libraries are installed.
2. **Data Loading and Preprocessing**: Follow the steps in the notebook to load and preprocess the data.
3. **Model Training and Evaluation**: Execute the provided code to train and evaluate the linear regression models.

## Requirements

- Python 3.x
- PySpark
- numpy
- matplotlib
- pandas

## Conclusion

This project demonstrates the application of linear regression to predict song release years using Spark, highlighting the importance of data preprocessing, model training, and hyperparameter optimization in a distributed computing environment.

## Acknowledgements

- Million Song Dataset for providing the data.
- UCI Machine Learning Repository for hosting the dataset.