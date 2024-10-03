# Income Prediction with Fairness Considerations
## Introduction

In this project, I built a machine learning model to predict whether an individual earns more than $50,000 a year using the UCI Adult Income Dataset. One of my main goals was to ensure that the model is fair and does not discriminate based on sensitive attributes like gender.

I noticed that machine learning models can sometimes be biased, which can lead to unfair decisions that affect people's lives. So, I wanted to explore how to detect such bias and apply techniques to mitigate it.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Steps](#project-steps)
   - [1. Loading and Exploring the Data](#1-loading-and-exploring-the-data)
   - [2. Data Preprocessing](#2-data-preprocessing)
   - [3. Training the Logistic Regression Model](#3-training-the-logistic-regression-model)
   - [4. Evaluating Model Fairness](#4-evaluating-model-fairness)
   - [5. Mitigating Bias Using Fairlearn](#5-mitigating-bias-using-fairlearn)
   - [Why I Used Fairlearn](#why-i-used-fairlearn)
   - [Understanding the Bias Mitigation Method](#understanding-the-bias-mitigation-method)
   - [6. Re-evaluating Fairness After Mitigation](#6-re-evaluating-fairness-after-mitigation)
- [Results](#results)
- [Conclusion](#conclusion)

## Dataset
The UCI Adult Income Dataset contains information about individuals from the 1994 Census database. It includes attributes like age, education, occupation, race, sex, and whether the individual's income exceeds $50,000.

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

## Project Steps

### 1. Loading and Exploring the Data
First, I loaded the dataset and took a look at its structure.

```python
import pandas as pd

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = [
   'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
   'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
   'hours_per_week', 'native_country', 'income'
]
data = pd.read_csv(url, names=columns)

# Display the first few rows
data.head()
```

### Visualizing Gender Distribution
I noticed that there might be an imbalance between males and females in the dataset, so I plotted the distribution.

```python
import matplotlib.pyplot as plt

# Plot the distribution of the 'sex' column
data['sex'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Sex in the Dataset')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
```

As we can see, there are more males than females in the dataset.

### 2. Data Preprocessing
I performed basic preprocessing to prepare the data for modeling.

- **Handling Missing Values:** Replaced ' ?' with NaN and dropped incomplete rows.
- **Encoding Categorical Variables:** Converted categorical variables into numerical format using one-hot encoding.

```python
# Handle missing values
data = data.replace(' ?', pd.NA).dropna()

# Convert categorical columns to dummy variables
data = pd.get_dummies(data, drop_first=True)
```

### 3. Training the Logistic Regression Model
I split the data into training and testing sets and trained a logistic regression model.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define features and target variable
X = data.drop('income_ >50K', axis=1)
y = data['income_ >50K']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy_original = accuracy_score(y_test, y_pred)
print(f'Original Accuracy: {accuracy_original:.4f}')
```

### 4. Evaluating Model Fairness
To check if the model was biased towards a particular gender, I used the Fairlearn library.

```python
from fairlearn.metrics import demographic_parity_difference

# Identify the sensitive feature (gender)
sex_feature = X_test['sex_ Male']  # 1 for Male, 0 for Female

# Calculate demographic parity difference
fairness_score_original = demographic_parity_difference(
   y_test, y_pred, sensitive_features=sex_feature
)
print(f'Original Fairness Score (Demographic Parity Difference for Gender): {fairness_score_original:.4f}')
```

The demographic parity difference was approximately 19%, indicating a bias towards one gender.

### 5. Mitigating Bias Using Fairlearn

#### Why I Used Fairlearn
I chose to use the Fairlearn library because it provides tools specifically designed to assess and mitigate bias in machine learning models. Since my initial model showed a significant bias towards one gender, I needed a way to reduce this bias while maintaining as much accuracy as possible.

Fairlearn offers algorithms that can adjust the model to make fairer predictions. It allowed me to:

- **Detect Bias:** By using metrics like demographic parity difference.
- **Mitigate Bias:** Through algorithms that enforce fairness constraints during model training.

#### Understanding the Bias Mitigation Method
I applied Fairlearn's Exponentiated Gradient method with a Demographic Parity constraint. Here's what that means in simple terms:

- **Exponentiated Gradient Method:**
  - Think of this as a smarter way to train the model.
  - It adjusts the model's predictions to find a good balance between accuracy and fairness.
  - It works by iteratively tweaking the model to reduce bias while trying to keep the accuracy high.

- **Demographic Parity Constraint:**
  - This is a rule that ensures the model gives positive outcomes (like predicting income > $50K) equally across different groups—in this case, males and females.
  - It aims to make the probability of a positive prediction the same for both genders.

In simple terms: I taught the model to treat men and women equally when predicting high incomes, making the model fairer.

```python
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# Initialize the mitigator
mitigator = ExponentiatedGradient(
   LogisticRegression(solver='liblinear'), constraints=DemographicParity()
)

# Fit the mitigator
mitigator.fit(X_train, y_train, sensitive_features=X_train['sex_ Male'])

# Predict with the mitigated model
y_pred_mitigated = mitigator.predict(X_test)

# Evaluate the mitigated model's accuracy
accuracy_mitigated = accuracy_score(y_test, y_pred_mitigated)
print(f'Accuracy after Mitigation: {accuracy_mitigated:.4f}')
```

### 6. Re-evaluating Fairness After Mitigation
I checked the fairness of the mitigated model.

```python
# Re-evaluate fairness
fairness_score_mitigated = demographic_parity_difference(
   y_test, y_pred_mitigated, sensitive_features=sex_feature
)
print(f'Fairness Score after Mitigation (Demographic Parity Difference for Gender): {fairness_score_mitigated:.4f}')

# Compare results
print("\nComparison:")
print(f"Accuracy Change: {accuracy_mitigated - accuracy_original:.4f}")
print(f"Fairness Improvement: {fairness_score_original - fairness_score_mitigated:.4f}")
```

## Results
- **Original Model Accuracy:** Approximately 84%
- **Original Fairness Score:** Approximately 19% demographic parity difference
- **Mitigated Model Accuracy:** Approximately 78%
- **Mitigated Fairness Score:** Approximately 0% demographic parity difference

## Conclusion
In this project, I aimed to build a predictive model for income levels while ensuring fairness across genders. The initial model, although accurate, showed a significant bias, favoring one gender over the other.

By applying the Exponentiated Gradient method with a Demographic Parity constraint from the Fairlearn library, I managed to:

- **Reduce the bias significantly:** The demographic parity difference dropped from 19% to almost 0%, indicating that the model's predictions are now fair across genders.
- **Accept a trade-off in accuracy:** The model's accuracy decreased from 84% to 78%. This trade-off is common when enforcing fairness constraints.

### Key Takeaways:
- **Why I Used Fairlearn:** To detect and mitigate bias in my model using specialized tools that help create fairer machine learning models.
- **Understanding Bias Mitigation:** The method adjusts the model to treat all groups equally, ensuring that sensitive attributes like gender do not unfairly influence the predictions.
- **Fairness vs. Accuracy Trade-off:** Ensuring fairness may come at the cost of reduced accuracy. It's essential to balance these aspects based on the application's requirements.
- **Importance of Bias Mitigation:** Detecting and mitigating bias is crucial in machine learning to prevent perpetuating societal inequalities.

