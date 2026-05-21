from LR_vector import gradient_descent,cost,prediction
import pandas as pd

data= pd.read_csv('data.csv')
#Convert DataFrame to Array for fastest multipication...
x=data.Hours_Studied.to_numpy()
y=data.Exam_Score.to_numpy()

mean=x.mean()
std=x.std()

x=(x-x.mean())/x.std() #Feature Scaling

m=0
b=0
L=0.01
epochs=5000

for i in range(epochs):
    m,b=gradient_descent(m,b,L,x,y)
print('m: ',m)
print('b: ',b)

hours=int(input('Enter hours studied: '))
print('Expected Marks: ', prediction(m,b,(hours-mean)/std))

