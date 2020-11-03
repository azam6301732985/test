#!/usr/bin/env python
# coding: utf-8

# # AUTHOR :- M.A.AZAM
# 
# TASK 1 :-Prediction Using Supervised Ml GRIP @ THE SPARKS FOUNDATION
# Linear Regression with Python Scikit Learn
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[3]:


# Importing all the libraries required in this notebook  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Step-1: Reading Data from Remote link

# In[6]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# # Step-2: Visualization the Input Data

# In[7]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# # Step-3: Preparing the Data
# ##The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[8]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# # Step-4: Splitting the data in to Training and Test sets.
# ##Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:
# 

# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# # Training the Algorithm
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# # Step-5: Plotting the line of Regression

# In[11]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # Step-6 Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.

# In[12]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# # Step-7: Comparing Actual vs Predicted Values
# 

# In[13]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[15]:


#Estimating the Train and Test
print('train values:',regressor.score(X_train,y_train))
print('test values:',regressor.score(X_test,y_test))


# In[19]:


#Plotting the line Graph to depict the difference between Actual and Predicted Values
df.plot(kind='line',figsize=(8,6))
plt.grid(which='major',linewidth='0.5',color='orange')
plt.grid(which='major',linewidth='0.5',color='blue')
plt.show()


# In[24]:


# testing with this data
hours = 9.25
test=np.array([hours])
test=test.reshape(-1,1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Step-8: Model Evaluation
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset.Here, we have chosen the Mean Absolute Error,Mean Squared Error,Root Mean Squared Error and R2

# In[26]:


from sklearn import metrics  
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:',metrics.r2_score(y_test,y_pred))


# # Conclusion
# 

# # R2 gives the score of model fit ,here R2  is actually a middle score for this model.

# # Thank you.
