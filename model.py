import pandas
import numpy
from sklearn.linear_model import LinearRegression

db=pandas.read_csv("salary.csv")
x = db['YearsExperience'].values.reshape(30,1)
y = db['Salary']
model = LinearRegression()
model.fit(x,y)
experience = int(input("Enter the experience using that we can predict the salary:"))
output = model.predict([[experience]])
print("Your Salary will be", output)
