import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('insurance.csv')
print(data.head())

""" Converting data """
lb = LabelEncoder()
data['sex'] = lb.fit_transform(data['sex'])
data['smoker'] = lb.fit_transform(data['smoker'])
data['region'] = lb.fit_transform(data['smoker'])
print(data.head())

"""Separating the data into x and y"""
x = data.iloc[:, :-1]
y = data['expenses']

"""Transformation"""
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)

"""Model Building"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
linear_model = LinearRegression().fit(x_train, y_train)
linear_model_score = linear_model.score(x_test, y_test)
print('the score for linear model is ', linear_model_score)
print('the coefficient value is ', linear_model.coef_)
print('the intercept value is ', linear_model.intercept_)
print('The predicted value is ', linear_model.predict([[19, 0, 27.9, 0, 1, 0]]))

pickle.dump(linear_model, open('insurance.pkl', 'wb'))
insurance = pickle.load(open('insurance.pkl', 'rb'))
