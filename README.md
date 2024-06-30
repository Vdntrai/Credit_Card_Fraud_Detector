

# Credit Card Fraud Detector

This project is focused on detecting fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, which is a common challenge in fraud detection tasks. The solution involves preprocessing the data, dealing with class imbalance using undersampling, and training a logistic regression model to classify transactions as fraudulent or genuine.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud is a significant issue that affects consumers and financial institutions globally. This project aims to build a predictive model that can identify fraudulent transactions from a dataset of credit card transactions.

## Dependencies

To run this project, you'll need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing

The dataset is loaded and initial exploration is performed to understand its structure. Key steps include:

1. Checking for null values and data types.
2. Understanding the distribution of the 'Class' variable.
3. Undersampling the majority class to address class imbalance.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('path/to/creditcard.csv')

# Data exploration
df.info()
df.describe()
```

## Model Training

A logistic regression model is used for training. The data is split into training and testing sets to evaluate the model's performance.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train-test split
X = df.drop(['Class'], axis=1)
Y = df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

## Evaluation

The model's performance is evaluated using accuracy metrics on both the training and test sets.

```python
# Predictions and accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(Y_train, train_pred)
print(f'Train Accuracy: {train_acc * 100:.2f}%')

test_pred = model.predict(X_test)
test_acc = accuracy_score(Y_test, test_pred)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
```

## Usage

To use the prediction system, input transaction data and the model will classify it as genuine or fraudulent.

```python
# Example prediction
data = (406, -2.312, 1.952, -1.610, 3.998, -0.522, -1.427, -2.537, 1.392, -2.770, -2.772, 3.202, -2.900, -0.595, -4.289, 0.390, -1.141, -2.830, -0.017, 0.417, 0.127, 0.517, -0.035, -0.465, 0.320, 0.045, 0.178, 0.261, -0.143, 0)
data_np = np.asarray(data).reshape(1, -1)
prediction = model.predict(data_np)
if prediction[0] == 0:
    print("Genuine Transaction")
else:
    print("Fraud Transaction")
```

## Results

The model achieves an accuracy of approximately X% on the training set and Y% on the test set. Further improvements can be made by experimenting with different sampling techniques and models.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

