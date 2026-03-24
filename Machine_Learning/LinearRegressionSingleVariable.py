import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df1 = pd.read_csv('/Users/Elisha/Desktop/PythonLeanring.csv')
df1

#Plotting the Graph
plt.scatter(df1.Area, df1.Price, color="red", s=100, marker="+")
plt.xlabel("Area(sqft)", fontsize=10)
plt.ylabel("Price($)" , fontsize=10)
plt.title("Trend of Housing Costs", fontsize=15, fontweight="bold")
plt.show()

#Regression
reg = linear_model.LinearRegression()
reg.fit(df1[['Area']], df1.Price)
area = float(input("Enter the area of house to find out cost: "))
print(reg.predict(pd.DataFrame([[area]], columns =["Area"])))


#Plot the line(which is a collection of points)
plt.scatter(df1.Area, df1.Price, color="red", s=100, marker="+")
plt.xlabel("Area(sqft)", fontsize=10)
plt.ylabel("Price($)" , fontsize=10)
plt.title("Trend of Housing Costs", fontsize=15, fontweight="bold")
plt.plot(df1.Area, reg.predict(df1[["Area"]]), color="blue")
plt.show()