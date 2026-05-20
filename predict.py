import pandas as pd
from linearregression import gradient_descent, prediction

data=pd.read_csv('data.csv')

m=0
b=0
L=0.0001
epochs=300

for i in range(epochs):
    m,b=gradient_descent(m,b,L,data)

a=int(input("Enter Hours Studied"))
result=prediction(m,b,a)

print("Expected Marks: ",result)