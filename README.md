# Gender Indetification by voice

## Executive Summary

This project explores the development of an automated gender classifier using machine learning techniques to improve data balance and mitigate biases in audio datasets. By accurately identifying the speaker's gender, this solution enhances the fairness and representativeness of datasets, benefiting various applications such as speech recognition and beyond. This classifier helps organisations improve model performance and adhere to ethical standards in AI development.

## Data Preprocessing
The data analysis process begins by exploring a ["Kaggle dataset"](https://www.kaggle.com/datasets/primaryobjects/voicegender/data) with 20 audio features and a gender label. The dataset is pre-cleaned and standardized, simplifying initial analysis. After loading and inspecting the dataset using Python and Pandas, the goal is to ensure it meets model training conditions. A benchmark accuracy of 50% is set by predicting the majority class ("male") in a balanced dataset, providing a baseline for evaluating any machine learning models developed.

## Initial Exploration
Using Python and Pandas, I will load and inspect the dataset to ensure it meets the necessary conditions for model training.
```python
kaggle_data = "voice.csv"
df = pd.read_csv(kaggle_data)
df.head()
```
![Screenshot: Initial_Exploration](screenshots/initial_exploration.png)

## Model Selection and Justification
For the modeling process, I chose XGBoost as the primary algorithm due to its proven performance and my familiarity with it. XGBoost has demonstrated high accuracy on this dataset, as highlighted by ["Primary Object"](https://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/) article, and I have experience tuning and optimizing it for various tasks. While CART models are also effective, my current expertise with them is limited. However, I plan to explore and implement CART models in future iterations to enhance the model and further develop my data science skills.

## Model Training
```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('voice.csv')

# Preprocess data
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # Convert labels to 0 and 1

# Split data into features and target
X = df.drop('label', axis=1)
y = df['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777)

# Define the model
params = {
    'objective': 'binary:logistic',
    'eta': 0.2,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'eval_metric': 'logloss'  # Focus on log loss as your evaluation metric
}

num_boost_round = 500

# Train the model
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain, num_boost_round)

# Predict on test set
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)
predictions = [1 if value >= 0.5 else 0 for value in y_pred]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```
This achieves an accuracy of 98.53%