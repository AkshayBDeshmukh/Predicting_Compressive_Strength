from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/cement_strength_reg/Concrete_Data.xls")

df.head()

df.columns

df.shape

"""- Cement (component 1)(kg in a m^3 mixture)

- Blast Furnace Slag (component 2)(kg in a m^3 mixture)

- Fly Ash (component 3)(kg in a m^3 mixture)

- Water  (component 4)(kg in a m^3 mixture)

- Superplasticizer (component 5)(kg in a m^3 mixture)

- Coarse Aggregate  (component 6)(kg in a m^3 mixture)

- Fine Aggregate (component 7)(kg in a m^3 mixture)

- Age (day)

- Concrete compressive strength(MPa, megapascals)
"""

#changing column name

col = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Concrete_compressive_strength ']

df.columns = col

df.head()

df.info()

df.describe()

#check for null values

df.isnull().sum()

df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

#plot the correlation matrix
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),cmap='Spectral', annot = True)

df.columns

"""# Scatter Plot Between Independent & Dependent Variable"""

col = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer',
       'Coarse_Aggregate', 'Fine_Aggregate', 'Age']

for i in col:
  plt.figure(figsize=(12,6))
  
  sns.scatterplot(df[i], df["Concrete_compressive_strength "])
  plt.title(f"Scatter Plot between {i} and concrete_compressive_strength")


from statsmodels.stats.outliers_influence import variance_inflation_factor

# the independent variables set
X = df[['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Superplasticizer', 'Fine_Aggregate', 'Age', 'Coarse_Aggregate', 'Water',]]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
						for i in range(len(X.columns))]

print(vif_data)

# the independent variables set
X = df[['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Superplasticizer', 'Fine_Aggregate', 'Age']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
						for i in range(len(X.columns))]

print(vif_data)

"""# Checking For Outliers """

for i in col:
  plt.figure(figsize=(8,4))
  sns.boxplot(df[i])
  plt.title(f"Box Plot of {i}")


for i in col:
  plt.figure(figsize=(8,4))
  sns.distplot(df[i])
  plt.title(f"Distribution Plot of {i}")


df.head()

X = df.drop("Concrete_compressive_strength ", axis = 1)
Y = df["Concrete_compressive_strength "]

from sklearn import preprocessing

# normalize the data attributes
X_scaled = preprocessing.normalize(X)

X_scaled

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled.head()

Y.head()


X.head()

from sklearn import preprocessing

# standardize the data attributes
X_scaled_std = preprocessing.scale(X)

X_scaled_std

X_scaled_std = pd.DataFrame(X_scaled_std, columns=X.columns)

X_scaled_std.head()

"""# Train-Test_split

### First i am going to take normalized data then will take standardized data.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.33, random_state=42)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)



from sklearn.linear_model import SGDRegressor 

lin_model = SGDRegressor()

#We fit our model with train data
lin_model.fit(X_train, Y_train)

# We use predict() to predict our values
lin_model_predictions = lin_model.predict(X_test)

# we check the coefficient of determination (R2 score) with score()
print(lin_model.score(X_test,Y_test))

#Mean Square Error

#We check the root mean square error (RMSE)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, lin_model_predictions)
rmse = np.sqrt(mse)
print(rmse)

#Actual vs Predicted Plot

fig, ax = plt.subplots()
ax.scatter(Y_test, lin_model_predictions)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()



from statsmodels.stats.stattools import durbin_watson

residuals = Y_test - lin_model_predictions
#perform Durbin-Watson test
durbin_watson(residuals)


#Q-Q plot for residuals

import statsmodels.api as sm
import pylab as py

residuals = Y_test - lin_model_predictions
sm.qqplot(residuals, line ='45')
py.show()

import warnings 

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

from sklearn.model_selection import GridSearchCV

# Grid search - this will take about some minute depending on system configuration
param_grid = {
    'alpha': 10.0 ** -np.arange(1,7),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    
}
grid_lin_model = GridSearchCV(lin_model, param_grid)
grid_lin_model.fit(X_train,Y_train)
print("Best Score:" +str(grid_lin_model.best_score_))

grid_lin_model.best_params_

grid_lin_model.best_estimator_

grid_model = grid_lin_model.best_estimator_

grid_predictions = grid_model.predict(X_test)

#coefficient of determination (R2 score) with score()
print(grid_model.score(X_test,Y_test))

#Mean Square Error

#We check the root mean square error (RMSE)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, grid_predictions)
rmse = np.sqrt(mse)
print(rmse)

#Actual vs Predicted Plot

fig, ax = plt.subplots()
ax.scatter(Y_test, grid_predictions)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#Durbin_watson_test
from statsmodels.stats.stattools import durbin_watson

residuals = Y_test - grid_predictions
#perform Durbin-Watson test
durbin_watson(residuals)

#Q-Q plot for residuals

import statsmodels.api as sm
import pylab as py

residuals = Y_test - grid_predictions
sm.qqplot(residuals, line ='45')
py.show()


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'alpha': 10.0 ** -np.arange(1,7),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    
}
random_lin_model = RandomizedSearchCV(lin_model, param_grid)
random_lin_model.fit(X_train,Y_train)
print("Best Score:" +str(random_lin_model.best_score_))

random_lin_model.best_params_

random_lin_model.best_estimator_

random_model = random_lin_model.best_estimator_

random_predictions = random_model.predict(X_test)

#coefficient of determination (R2 score) with score()
print(random_model.score(X_test,Y_test))

#Mean Square Error

#We check the root mean square error (RMSE)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, random_predictions)
rmse = np.sqrt(mse)
print(rmse)

#Actual vs Predicted Plot

fig, ax = plt.subplots()
ax.scatter(Y_test, random_predictions)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#Durbin_watson_test
from statsmodels.stats.stattools import durbin_watson

residuals = Y_test - random_predictions
#perform Durbin-Watson test
durbin_watson(residuals)

#Q-Q plot for residuals

import statsmodels.api as sm
import pylab as py

residuals = Y_test - random_predictions
sm.qqplot(residuals, line ='45')
py.show()



X_scaled.head()

input = X_scaled.copy()

# Dropping water and Coarse_Aggregate feature 

input.drop("Water", axis = 1, inplace = True)
input.drop("Coarse_Aggregate", axis = 1, inplace = True)

input.head()

Y.head()

# Train-Test-Split
X_train, X_test, Y_train, Y_test = train_test_split(input, Y, test_size=0.20, random_state=101)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Creating linear modle with the help of RandomizedSearchCV

param_grid = {
    'alpha': 10.0 ** -np.arange(1,7),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling']}
    
random_lin_model = RandomizedSearchCV(lin_model, param_grid)
random_lin_model.fit(X_train,Y_train)
print("Best Score:" +str(random_lin_model.best_score_))

random_lin_model.best_params_

random_lin_model.best_estimator_

random_model = random_lin_model.best_estimator_

random_predictions = random_model.predict(X_test)

#R2 score
random_model.score(X_test,Y_test)

#Mean Square Error

mse = mean_squared_error(Y_test, random_predictions)
rmse = np.sqrt(mse)
print(rmse)

#Actual vs Predicted Plot

fig, ax = plt.subplots()
ax.scatter(Y_test, random_predictions)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#Durbin_watson_test

residuals = Y_test - random_predictions
#perform Durbin-Watson test
durbin_watson(residuals)

#Q-Q plot for residuals

residuals = Y_test - random_predictions
sm.qqplot(residuals, line ='45')
py.show()



import xgboost as xg

# Instantiation
xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)
  
# Fitting the model
xgb_r.fit(X_train,Y_train)
  
# Predict the model
pred = xgb_r.predict(X_test)
  
# RMSE Computation
rmse = np.sqrt(mean_squared_error(Y_test, pred))
print("RMSE : % f" %(rmse))

#R2 Score
xgb_r.score(X_test,Y_test)

#Actual vs Predicted Plot

fig, ax = plt.subplots()
ax.scatter(Y_test, pred)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#Durbin_watson_test

residuals = Y_test - pred
#perform Durbin-Watson test
durbin_watson(residuals)

#Q-Q plot for residuals

residuals = Y_test - pred
sm.qqplot(residuals, line ='45')
py.show()



from sklearn.model_selection import RandomizedSearchCV
import xgboost as xg

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}


regressor = xg.XGBRegressor() 


random_search=RandomizedSearchCV(regressor,param_distributions=params,n_iter=5,scoring='neg_root_mean_squared_error',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train,Y_train)

random_search.best_params_

random_search.best_estimator_

random_model = random_search.best_estimator_

random_predictions = random_model.predict(X_test)

#R2 score
random_model.score(X_test,Y_test)

#Mean Square Error

mse = mean_squared_error(Y_test, random_predictions)
rmse = np.sqrt(mse)
print(rmse)

#Actual vs Predicted Plot

fig, ax = plt.subplots()
ax.scatter(Y_test, random_predictions)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#Durbin_watson_test

residuals = Y_test - random_predictions
#perform Durbin-Watson test
durbin_watson(residuals)

#Q-Q plot for residuals

residuals = Y_test - random_predictions
sm.qqplot(residuals, line ='45')
py.show()

X_test.head()

X_train.head()



X = df.drop("Concrete_compressive_strength ", axis = 1)
Y = df["Concrete_compressive_strength "]

X

from sklearn import preprocessing

# normalize the data attributes
X_scaled = preprocessing.normalize(X)

X_scaled

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=42)

#XGBoost Model

from sklearn.model_selection import RandomizedSearchCV
import xgboost as xg

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}


regressor = xg.XGBRegressor() 


random_search=RandomizedSearchCV(regressor,param_distributions=params,n_iter=5,scoring='neg_root_mean_squared_error',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train,Y_train)

random_search.best_params_

random_search.best_estimator_

random_model = random_search.best_estimator_

random_predictions = random_model.predict(X_test)

#R2 Score

random_model.score(X_test,Y_test)

#Mean Square Error

mse = mean_squared_error(Y_test, random_predictions)
rmse = np.sqrt(mse)
print(rmse)

#Actual vs Predicted Plot

fig, ax = plt.subplots()
ax.scatter(Y_test, random_predictions)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

#Durbin_watson_test

residuals = Y_test - random_predictions
#perform Durbin-Watson test
durbin_watson(residuals)

"""# Creating pickle file for random_model"""

### Create a Pickle file using serialization 
import pickle
pickle_out = open("regressor.pkl","wb")
pickle.dump(random_model, pickle_out)
pickle_out.close()

random_model.predict([[0.175,0.00,0.00,0.00,0.65,0.0028,0,0]])
