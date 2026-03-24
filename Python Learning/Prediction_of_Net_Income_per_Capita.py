import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

df = pd.read_csv("/Users/Elisha/Desktop/Python learning/canada_per_capita_income.csv")

#Show graph
plt.scatter(df['year'], df['per_capita_income_(USD)'])
plt.xlabel("Year" , fontweight="bold")
plt.ylabel("Per Capita Income(USD)" , fontweight="bold")
plt.title("Net Income Per Capita", fontweight="bold", fontsize=15)
plt.show()

#ML regression(Ridge)
R_Reg = make_pipeline(PolynomialFeatures(degree=2), linear_model.Ridge())
R_Reg.fit(df[["year"]], df["per_capita_income_(USD)"])
plt.scatter(df['year'], df['per_capita_income_(USD)'])
plt.xlabel("Year" , fontweight="bold")
plt.ylabel("Per Capita Income(USD)" , fontweight="bold")
plt.title("Net Income Per Capita", fontweight="bold", fontsize=15)
plt.plot( df["year"], R_Reg.predict(df[["year"]]), color = "Blue" )
plt.show()

while True:
    Year = input("Enter a year ( or 'Quit' to stop) :")
    if Year.lower() == "quit":
        break
    Year = float(Year)
    PR = R_Reg.predict(pd.DataFrame([[Year]], columns=["year"]))
    print(f"The Net Income for the {int(Year)} is ${PR[0]:,.2f}")