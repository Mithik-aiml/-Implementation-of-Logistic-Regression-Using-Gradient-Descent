# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.
<table>
<tr>
<th>


## Program:
</th>
<th>

## Output:
</th>
</tr>
<tr>
<td width=40%>
  
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
df = pd.read_csv('Placement_Data (1).csv')
df.head()
```

</td> 
<td>

<img width="1305" height="252" alt="image" src="https://github.com/user-attachments/assets/d0c7a137-4350-43da-b2cf-fd000288e2c2" />

</td>
</tr> 
</table>
<table>
<tr>
<td width=40%>
  
```Python
df = df.drop('sl_no',axis=1) 
df = df.drop('salary',axis=1) 
A=["gender","ssc_b","hsc_b","degree_t",
   "workex","specialisation","status","hsc_s"]
for i in A:
    df[i]=df[i].astype('category')
df.dtypes
```
</td> 
<td>

<img width="279" height="586" alt="image" src="https://github.com/user-attachments/assets/d7459a26-0db2-405a-957d-621471a4012e" />


</td>
</tr> 
</table>
<table>
<tr>
<td width=40%>
  
```Python
for j in A:
    df[j]=df[j].cat.codes
df
```

</td> 
<td>

<img width="1111" height="507" alt="image" src="https://github.com/user-attachments/assets/85b670c5-b627-4de4-a554-0cc004eb4d64" />


</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
Y
```

</td> 
<td>

<img width="719" height="224" alt="image" src="https://github.com/user-attachments/assets/8ce771d7-c1db-446a-aebd-abb62025e186" />


</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
# Define the gradient descent algorithm.
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
# Train the model.
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
# Make predictions.
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
# Evaluate the model.
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy) 
```

</td> 
<td>

<img width="275" height="15" alt="image" src="https://github.com/user-attachments/assets/736bf933-ea8c-4afd-88fc-31d0343fcf3f" />


</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
print(y_pred)
```

</td> 
<td>

<img width="727" height="137" alt="image" src="https://github.com/user-attachments/assets/195ce132-bf64-4c65-af42-719e49edb2bc" />


</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
print(Y)
```

</td> 
<td>

<img width="761" height="139" alt="image" src="https://github.com/user-attachments/assets/06d18070-9645-495d-8614-065f77df5d37" />

</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])        
y_prednew = predict(theta, xnew)
print(y_prednew)
```

</td> 
<td>

<img width="55" height="30" alt="image" src="https://github.com/user-attachments/assets/497f416b-5fff-4e75-b3aa-9b8d8712de6f" />



</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])        
y_prednew = predict(theta, xnew)
print(y_prednew)
```
</td> 
<td>
<img width="55" height="30" alt="image" src="https://github.com/user-attachments/assets/b8c499aa-0ad6-4e23-b9ec-094e2609f8f9" />

</td>
</tr> 
</table>



```
Developed : G.Mithik jain
Reg no:212224240087
```
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

