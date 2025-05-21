# Instacart Reorder Prediction – Machine Learning Classification Project

This project focuses on predicting whether a customer will reorder a product in their next purchase, based on historical transaction data from the Instacart Market Basket Analysis dataset. The problem is framed as a multi-class classification task (not reordered, reordered, newly ordered).

## Project Objective

The goal was to build a model that forecasts product reordering behavior to help e-commerce platforms optimize inventory management, reduce stock shortages, and improve personalized recommendations.

## Dataset Overview

- Source: This project is based on the Instacart Market Basket Analysis dataset, which was originally released in a Kaggle competition by Instacart in 2017.
While the original competition page is no longer available, the dataset is still accessible through the Kaggle community:
👉 [Instacart Market Basket Analysis – Kaggle dataset by psparks](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
- Size: ~20.5 million rows
- Structure: multiple tables (orders, products, aisles, departments, prior orders, train orders)

## Key Challenges

- Sequential nature of data (temporal dependencies)
- Risk of data leakage
- Multi-class target with class imbalance
- Large volume and sparse structure of features

## Workflow Summary

- Data cleaning and preparation
- Exploratory Data Analysis (EDA)
- Feature engineering based on user behavior, time gaps, product categories
- Model development using:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (best-performing)
  - CatBoost
  - GRU (tested on subsample)
- Model evaluation with F1-score and confusion matrix
- SHAP value analysis for interpretability

## Final Results (XGBoost)

| Class | F1-score |
|-------|----------|
| 0     | 0.77     |
| 1     | 0.82     |
| 2     | 0.26     |
| **Macro avg** | **0.61** |

XGBoost outperformed all other models in terms of speed and predictive performance, though class 2 (new product) remains a modeling challenge due to its low support.

## Repository Structure

- `instacart_notebook.ipynb`: full notebook with code and explanation
- `charts/`: model evaluation visuals (SHAP, feature importance, class distribution)
- `presentation_ml_instacart.pdf`: summary of the project
- `requirements.txt`: list of libraries used

## Next Steps

- Optimize GRU model structure for full sequential training
- Explore separate binary classifier for class 2
- Further hyperparameter tuning and feature selection

For more details, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/karina-oborska-balkowiec/)
