import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read('data.csv')

def cost(m,b,points):
    error=0
    x=np.array(points.Hours_Studied)
    y=np.array(points.Exam_Score)

    error=np.mean(y-(m*x+b))**2

    return error

def gradient_descent(m_now,b_now,points,L):
    m_grad=0
    b_grad=0

    x=np.array(points.Hours_Studied)
    y=np.array(points.Exam_Score)

    y_pred=m_now*x+b_now

    m_grad=(-2)*np.mean(x*(y-y_pred))
    b_grad=(-2)*np.mean((y-y_pred))

    m=m_now-m_grad*L
    b=b_now-b_grad*L

    return m,b

def prediction(m,b,hours):
    return m*hours+b

