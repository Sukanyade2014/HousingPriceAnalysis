#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('/Users/sukanyade/Downloads/housing.csv')
print(df.head())


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


#Check for missing values:
df.isna().sum()


# In[6]:


#Convert ocean_proximity to numeric since the other features are nemeric:
df['ocean_proximity'].value_counts()
mappings = {"<1H OCEAN" : 1 , "INLAND" : 2 , "NEAR OCEAN" : 3,  "ISLAND" : 4, "NEAR BAY" : 5}
df = df.replace({'ocean_proximity' : mappings})
print(df.head())



# In[7]:


#Total Bedroom has missing values. Filling it with median values of the feature:
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
iter_imputer = IterativeImputer()
housing_imputed = iter_imputer.fit_transform(df)
housing_imputed = pd.DataFrame(housing_imputed , columns = df.columns)


# In[8]:


# Running a simple validation to check if the replacement function is valid for the data 
df1 = housing_imputed[housing_imputed['total_rooms'] < housing_imputed['total_bedrooms']]
print(df1.head())


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
#Pairplot to find the corelation between the features:
sns.pairplot(housing_imputed, vars = ['housing_median_age' , 'population' , 'total_rooms' , 'total_bedrooms' , 'households'  ,'median_income' , 'median_house_value' ,'ocean_proximity'  ])


# In[10]:


# The graph shows that Population, Total_Bedrooms, Household and total_Rooms are highly corelated
#Creating new features to check for their corelations:

housing_imputed['rooms_per_household'] = housing_imputed['total_rooms']/housing_imputed['households']
housing_imputed['bedroom_per_room'] = housing_imputed['total_bedrooms']/housing_imputed['total_rooms']
housing_imputed['population_per_household'] = housing_imputed['population']/housing_imputed['households']


# In[11]:


sns.pairplot(housing_imputed, vars = ['rooms_per_household' , 'bedroom_per_room' , 'population_per_household' , 'housing_median_age' , 'population' , 'total_rooms' , 'total_bedrooms' , 'households'  ,'median_income' , 'median_house_value' ,'ocean_proximity'  ])


# In[12]:


from sklearn.preprocessing import PolynomialFeatures, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression

x = housing_imputed.drop('median_house_value' , axis=1)
y = housing_imputed['median_house_value']

xtrain, xtest, ytrain, ytest = train_test_split(x,y , test_size = .20 , random_state=10)
pipe = Pipeline((
("it" , IterativeImputer(estimator = LinearRegression())),
    ("pt" , PowerTransformer()),
    ("sc" , StandardScaler()),
    ("lr" , LinearRegression()),
))
pipe.fit(xtrain, ytrain)
print(pipe.score(xtrain, ytrain))
print(pipe.score(xtest , ytest))


# In[13]:


#Using DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
x = housing_imputed.drop('median_house_value' , axis=1)
y = housing_imputed['median_house_value']

xtrain, xtest, ytrain, ytest = train_test_split(x,y , test_size = .20 , random_state=10)
pipe = Pipeline((
("it" , IterativeImputer(estimator = LinearRegression())),
    ("pt" , PowerTransformer()),
    ("sc" , StandardScaler()),
    ("lr" , DecisionTreeRegressor()),
))
pipe.fit(xtrain, ytrain)
print(pipe.score(xtrain, ytrain))
print(pipe.score(xtest , ytest))


# In[14]:


#RandomForestModel
from sklearn.ensemble import RandomForestRegressor
x = housing_imputed.drop('median_house_value' , axis=1)
y = housing_imputed['median_house_value']

xtrain, xtest, ytrain, ytest = train_test_split(x,y , test_size = .20 , random_state=10)
pipe = Pipeline((
("it" , IterativeImputer(estimator = LinearRegression())),
    ("pt" , PowerTransformer()),
    ("sc" , StandardScaler()),
    ("lr" , RandomForestRegressor()),
))
pipe.fit(xtrain, ytrain)
print(pipe.score(xtrain, ytrain))
print(pipe.score(xtest , ytest))


# In[16]:


#RandomForestModel has the best score.
# Calculate cross validation score for the RandomForestModel

from sklearn.model_selection import cross_val_score
score_rf = cross_val_score(pipe, xtrain, ytrain, cv=5, scoring='r2'
                          )
print(score_rf)


# In[17]:


#Estimate Confidence Interval
import scipy.stats as stats
n=50
xbar= np.mean(score_rf)
s = np.std(score_rf)
se = s/np.sqrt(n)
stats.t.interval(0.95, df= n-1, loc=xbar, scale=se)


# In[ ]:




