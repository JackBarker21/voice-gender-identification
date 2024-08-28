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