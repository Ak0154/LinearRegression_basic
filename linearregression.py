import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('data.csv')

# plt.scatter(data.Hours_Studied,data.Exam_Score)
# plt.show()

def gradient_descent(m_now,b_now,L,points):
    m_grad=0
    b_grad=0

    n=len(points)

    for i in range(n):
        x=points.iloc[i].Hours_Studied
        y=points.iloc[i].Exam_Score

        #DIRECTION FOR THE GRAPH

        m_grad+=-2/n * x*(y-(m_now*x+b_now))
        b_grad+=-2/n*(y-(m_now*x+b_now))

        #The final values of m to minimize Error where L=Learning Rate

    m=m_now-m_grad*L
    b=b_now-b_grad*L

    return m,b

m=0
b=0
L=0.0001
epochs=300   #No. of iterations

for i in range(epochs):
    m,b=gradient_descent(m,b,L,data)

print (m,b)

plt.scatter(data.Hours_Studied,data.Exam_Score)
plt.plot(list(range(1,16)), [m*x+b for x in range(1,16)], color='red')
plt.show()


