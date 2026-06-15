# wine dataset --> data analysis --> data pre processing --> train test split --> Random forest modal --> trained random modal <-- new data --> prediction

# importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# data collection
# loading the data sets to pandas data frame
wine_dataset = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\winequality-red.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(wine_dataset.head())
print(wine_dataset.shape)

# checking for null values
print(wine_dataset.isnull().sum())

# data analysis abd visulaization
print(wine_dataset.describe())

# number of values for each quality
sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.show()

# volatile acidity vs Quality
plot = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
plt.show()  # inversely proportional

# citric acid  vs Quality
plot2 = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)
plt.show()  # directly proportional

# correlation
# positive correlation
# negative correlation
correlation = wine_dataset.corr()
# constructing the heat map tp understand the correlation between columns
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={"size": 8}, cmap="Blues")
plt.show()

# Data pre processing
# separate the data and lebal
x = wine_dataset.drop('quality', axis=1)
print(x)
y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
print(y)

# Train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
print(y_train.shape)
print(y_test.shape)

# modal training
# random forest classification
modal = RandomForestClassifier()
modal.fit(x_train, y_train)

# modal Evaluation
# accuracy score
# accuracy on test data
x_test_pridiction = modal.predict(x_test)
test_data_accuracy = accuracy_score(x_test_pridiction, y_test)

print("Accuracy on test data ",test_data_accuracy)

# build predictive system
input_data = (7.5,0.52,0.16,1.9,0.085,12.0,35.0,0.9968,3.38,0.62,9.5)
# changing data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the data as we are predicting only the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = modal.predict(input_data_reshaped)
print(prediction)

if prediction == 1:
    print("IT IS GOOD QUALITY WINE")
else:
    print("IT IS NOT GOOD QUALITY WINE")

