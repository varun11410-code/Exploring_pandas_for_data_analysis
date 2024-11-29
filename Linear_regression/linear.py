import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import random

#load dataset
data = pd.read_csv("cardiovascular.csv")

#print dataset columns
print(data.columns)

#print the correlation between the columns
print(data.corr().to_string())

#Define features and target variable
X=data[['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
Y=data['target']
print(X)
print(Y)

#plot the data for easy visualization
sns.scatterplot(data=data)
plt.plot(X,Y)
plt.show()

# Split the dataset into training and testing sets
random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)

# Initialize and train the Linear Regression model
model = LinearRegression()   
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)
Y_pred_rounded = np.round(Y_pred)

# Print the accuracy score
accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"Accuracy Score: {accuracy * 100:.2f}%")