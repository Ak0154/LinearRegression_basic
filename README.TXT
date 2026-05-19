Linear regression is a machine learning/statistical method used to predict a continuous value by finding the best-fit straight line between input variables and an output variable.

Mathematically:

y=mx+b

where:

y → predicted output (dependent variable)
x → input feature (independent variable)
m → slope (how much y changes when x changes)
b → intercept (value of y when x=0)

Example:

Suppose we have:

Hours Studied	Exam Score
2	45
4	58
6	70
8	82

Linear regression finds a line like:

Exam_Score=6(Hours_Studied)+33

Interpretation:

Each additional hour studied increases score by about 6 marks
If study hours were 0, predicted score would be about 33

Impementation using pandas and matplotlib

Replace the name of the file with your csv file.
note: This program only works for 1 feature i.e., from 1 x, we predict y.

The gradient descent funtion calculate the best line by continuously adjusting the value of 
m and b.

m_now → current slope
b_now → current intercept
L → learning rate
points → dataset