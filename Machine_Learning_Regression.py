
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import pickle

# In[2]:


df = pd.read_csv('Salary_Data.csv')


# In[4]:


x = df.iloc[:,:-1].values


# In[5]:


y = df.iloc[:,1].values


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)


# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
regressor = LinearRegression()


# In[15]:


regressor.fit(x_train,y_train)


# In[16]:


regressor.intercept_


# In[17]:


regressor.coef_


# In[18]:


y_pred = regressor.predict(x_test)


# In[19]:


y_pred


# In[21]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[25]:


dataset = pd.read_csv('50_Startups.csv')


# In[27]:


# print(dataset.head())







# In[45]:


x_ = dataset.iloc[:,:-1].values


# In[46]:


y_ = dataset.iloc[:,4].values


# In[47]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

# In[48]:


x_[:,3] = labelencoder.fit_transform(x_[:,3])


# In[49]:

ct = ColumnTransformer([("", OneHotEncoder(), [3])], remainder = 'passthrough')

# onehotencoder = OneHotEncoder(categorical_features=[3])
# x_ = onehotencoder.fit_transform(x_).toarray()
x_ = ct.fit_transform(x_)

# In[52]:


x_ = x_[:,1:]


# In[64]:


from sklearn.model_selection import train_test_split
x__train, x__test, y__train, y__test = train_test_split(x_,y_, test_size = 0.2, random_state = 0)


# In[65]:


from sklearn.linear_model import LinearRegression
regression_ = LinearRegression()


# In[66]:


regression_.fit(x__train, y__train)

# with open('test.pkl', 'wb') as file:
#     pickle.dump(regression_, file)
# pickle.dumps(regression_,open("model.pkl","wb"))
# In[67]:


regression_.intercept_


# In[68]:


regression_.coef_


# In[69]:


y__pred = regression_.predict(x__test)

print()

# In[70]:
print(regression_.predict([[0.0,1.0,105349.2,156897.8,571784.1]]))

y__pred


# In[71]:


from sklearn.metrics import r2_score
print(r2_score(y__test,y__pred))



