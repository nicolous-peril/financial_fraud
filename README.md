# Financial Fraud Detection Project

A machine learning project focused on detecting fraudulent financial transactions using a full end-to-end pipeline: from raw data to tuned models.

## Project Overview

In this project, I worked on detecting fraudulent financial transactions using machine learning. Since fraud cases are rare compared to normal transactions, the main challenge was handling an imbalanced dataset and still being able to correctly identify fraud.

I built models that can catch as many fraudulent transactions as possible, while minimizing false alarms.

## Problem Context

This dataset is **highly imbalanced** — most transactions are not fraud.

That means:
* Accuracy alone is misleading
* Need better metrics

So the focus is on:
**Precision + Recall → F1 Score**

## Project Structure
```
financial_fraud/
├── code/
│   ├── 01_explore.ipynb
│   ├── 02_transform.ipynb
│   └── 03_model.ipynb
│
├── data/
│   ├── caishen_bank_transactions.csv
│   └── cleaned_fraud_data.csv
│
├── docs/
│   ├── images/
│   ├── report/
│
├── .gitignore
└── README.md
```

## Exploratory Data Analysis

### Class Distribution

**Key insights:**
* Fraud cases are extremely rare
* Certain transaction types are more suspicious
* Large amounts show stronger fraud signals
* Balance changes reveal hidden patterns

## Data Cleaning & Preprocessing

What I did:
* Dropped unnecessary columns
* Focused on meaningful financial features
* Cleaned inconsistencies in balances
* Saved processed dataset for modeling

## Model Development

**Models Used:**
* Random Forest
* Gradient Boosting

Why:
* Great for tabular data
* Capture complex patterns
* Handle noise well

## Hyperparameter Tuning

Used:
**RandomizedSearchcv**

Reasons:
* Faster than GridSearch, especially for large datasets
* Samples different parameter combinations efficiently
* Still finds strong model performance without testing everything

## Model Results

### Final F1 Scores
| Model                | F1 Score | Notes                         |
|---------------------|----------|-------------------------------|
| Random Forest (Base)| 0.88     | Strong baseline performance   |
| Random Forest (Tuned)| 0.80    | Slight decrease after tuning  |
| Gradient Boosting (Base)| 0.65 | Lower recall initially        |
| Gradient Boosting (Tuned)| 0.71| Improved after tuning         |

