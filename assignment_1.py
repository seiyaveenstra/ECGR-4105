import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ex =pd.read_csv('https://raw.githubusercontent.com/seiyaveenstra/ECGR-4105/main/D3.csv')

m = ex.Y.size
X0 = np.ones((m,1))
X0 = pd.DataFrame(X0)
ex.insert(0,"intercept",X0)
print(ex)
#first explanatory variable
x1 = ex.loc[:,['intercept','X1']]
y = ex.Y
theta = np.zeros(2)
iterations = 1500
alpha = 0.1

def cost(x,y,theta):
    m = len(x)
    predictions = x.dot(theta)
    errors = np.subtract(predictions,y)
    sqrErrors = np.square(errors)
    J = 1/(2*m)*np.sum(sqrErrors)
    return J

def gd(x,y,theta,alpha,iterations):
    costHis = np.zeros(iterations)
    for i in range(iterations):
        pred = x.dot(theta)
        errors = np.subtract(pred,y)
        sumDelta = (alpha/m)*x.transpose().dot(errors)
        theta = theta -sumDelta
        costHis[i] = cost(x,y,theta)
    return theta, costHis

xarray = x1.loc[:,['intercept','X1']].values
yarray = y.values
theta,costHis = gd(xarray,yarray,theta,alpha,iterations)
print('theta = ',theta)
print('cost history = ',costHis)

plt.scatter(x1.X1,y,label = 'Training data')
plt.plot(x1.X1,x1.dot(theta),label = 'linear regression')
plt.grid()
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('linear regression graph')
plt.legend()
plt.show()
#second expalnatory variable
plt.figure()
x2 = ex.loc[:,['intercept','X2']]
xarray = x2.loc[:,['intercept','X2']].values
yarray = y.values
theta,costHis = gd(xarray,yarray,theta,alpha,iterations)
print('theta = ',theta)
print('cost history = ',costHis)

plt.scatter(x2.X2,y,label = 'Training data')
plt.plot(x2.X2,x2.dot(theta),label = 'linear regression')
plt.grid()
plt.xlabel('X2')
plt.ylabel('Y')
plt.title('linear regression graph')
plt.legend()
plt.show()

#third expalnatory variable
plt.figure()
x3 = ex.loc[:,['intercept','X3']]
xarray = x3.loc[:,['intercept','X3']].values
yarray = y.values
theta,costHis = gd(xarray,yarray,theta,alpha,iterations)
print('theta = ',theta)
print('cost history = ',costHis)

plt.scatter(x3.X3,y,label = 'Training data')
plt.plot(x3.X3,x3.dot(theta),label = 'linear regression')
plt.grid()
plt.xlabel('X3')
plt.ylabel('Y')
plt.title('linear regression graph')
plt.legend()
plt.show()