# Linear Regression from Scratch

Linear Regression is a machine learning/statistical method used to predict a continuous value by finding the **best-fit straight line** between input variables and an output variable.

Mathematically:
                    [y = mx + b]

where:

* **y** → predicted output (dependent variable)
* **x** → input feature (independent variable)
* **m** → slope (how much *y* changes when *x* changes)
* **b** → intercept (value of *y* when *x = 0*)

## Example

Suppose we have the following dataset:

| Hours Studied | Exam Score |
| ------------- | ---------- |
| 2             | 45         |
| 4             | 58         |
| 6             | 70         |
| 8             | 82         |

Linear regression finds a line similar to:

[
Exam_Score = 6(Hours_Studied) + 33
]

### Interpretation

* Each additional hour studied increases the score by approximately **6 marks**
* If study hours are **0**, the predicted score would be approximately **33**

## Implementation using Pandas and Matplotlib

Replace the file name with your own CSV file.

**Note:**
This implementation currently works only for **single-feature linear regression**, meaning we use one input variable (**x**) to predict the output (**y**).

## Gradient Descent

The `gradient_descent()` function calculates the best-fit line by continuously adjusting the values of **m** and **b** to minimize prediction error.

Parameters:

* `m_now` → current slope
* `b_now` → current intercept
* `L` → learning rate
* `points` → dataset

The process works as follows:

1. Calculate prediction error
2. Compute gradients for `m` and `b`
3. Update `m` and `b`
4. Repeat for multiple epochs until the error is minimized
