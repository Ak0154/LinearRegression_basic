import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('data.csv')

# plt.scatter(data.Hours_Studied,data.Exam_Score)
# plt.show()

def Mean_Squarred_error(m,b,points):
    E=0
    n=len(points)
    for i in range(n):
        x=points.iloc[i].Hours_Studied
        y=points.iloc[i].Exam_Score

        E+=(y-(m*x+b))**2

    return E/n

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

#Pass Values

m=0
b=0
L=0.00001
epochs=300 #No. of iterations

epoch_plot=[]  
costs=[]
for i in range(epochs):
    m,b=gradient_descent(m,b,L,data)
    cost=Mean_Squarred_error(m,b,data)
    costs.append(cost)
    if(i%50==0):   
        # epoch_plot.append(i)
        print(
            f"Epoch:{i}, "
            f"m:{m:.3f}, "
            f"b:{b:.3f}, "
            f"Cost:{cost:.3f}"
        )



plt.scatter(data.Hours_Studied,data.Exam_Score)
# x=data.Hours_Studied
# plt.plot(x, [m*i+b for i in x], color='red')
# plt.plot(range(epochs),costs, color='green')
plt.show()


