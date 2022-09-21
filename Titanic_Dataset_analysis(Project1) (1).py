#!/usr/bin/env python
# coding: utf-8

# # Titanic Project

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Import Titanic csv file

# In[8]:


titanic = pd.read_csv('C:\\Users\\arunc\\Desktop\\data cleaning\\Titanic_dataset.csv')
titanic.head()


# In[9]:


titanic.shape


# # Finding and Filling the missing values

# In[10]:


titanic.isnull().sum()


# In[11]:


sns.heatmap(titanic.isnull(),yticklabels=False);


# In[12]:


titanic['Pclass'].unique()


# In[13]:


titanic.drop('Cabin',axis=1,inplace=True)
titanic.head()


# In[15]:


mean_age = titanic.groupby('Pclass').mean()['Age']


# In[16]:


mean_age


# In[17]:


#fillna(titanic['Age'].mean())


# In[18]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 41
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[19]:


titanic['Age'] = titanic[['Age','Pclass']].apply(impute_age,axis=1)


# In[20]:


sns.heatmap(titanic.isnull(),yticklabels=False);


# In[21]:


titanic.isnull().sum()


# In[23]:


titanic['Fare']= titanic['Fare'].fillna(titanic['Fare'].mean())


# In[24]:


sns.heatmap(titanic.isnull(),yticklabels=False);


# In[25]:


titanic.dropna(inplace=True)


# In[27]:


titanic.isnull().sum()


# # Converting Category data to numerical data

# In[28]:


titanic.info()


# In[ ]:


# Name,Sex,Ticket,Embarked is categorical data need to convert to numerical data


# In[29]:


Sex = pd.get_dummies(titanic['Sex'])
Sex.head()


# In[30]:


Sex = pd.get_dummies(titanic['Sex'],drop_first = True)
Sex.head()


# In[32]:


Embark = pd.get_dummies(titanic['Embarked'])
Embark.head()


# In[33]:


Embark = pd.get_dummies(titanic['Embarked'],drop_first = True)
Embark.head()


# In[34]:


titanic.head()


# In[35]:


titanic.drop(['Sex','Embarked','Name','PassengerId','Ticket'],axis=1,inplace=True)


# In[36]:


titanic.head()


# In[37]:


titanic = pd.concat([titanic,Sex,Embark],axis=1)
titanic.head()


# In[38]:


titanic.info()


# # Visualize and Corelate the Dataset

# In[ ]:


###As our dataset was independent and was dependent on other variables so to find out any relation between and 2 variables 
#or more we have to convert our Object data into Numerical data.And now we can easily plot them and find some relation


# In[39]:


titanic.head()


# In[44]:


plt.figure(figsize=(8,4))
plt.xlabel('Age')
titanic['Age'].plot.hist(edgecolor='b').autoscale(enable=True,axis='both',tight=True);


# In[ ]:


#To find out relationship between Pclass and SibSp and now we can find out who are the  maximum and minimum no. of Pclass people
# with thier number of Siblings and Parents.Hence People with 0 SibSp are of Pclass=3 and maximum with 8 SibSp of 
#same Plcass=3 only.

#And we can find other relations too!


# In[41]:


figure = plt.figure(figsize=(10,5));
plt.title('Titanic');
sns.countplot(x='SibSp',hue='Pclass',data=titanic);
plt.xlabel('SibSp',fontsize=14);
plt.ylabel('count',fontsize=14);
plt.yticks(fontsize=14);


# In[ ]:


# Here we Can clearly depict the average age of all Pclass.
#Pclass1. has people with average people between 40-45
#Plcass2. has people with average people between 25-30
#Pclass3. has people with average people between 22-28
###Hence we can also find some Outliers in both Pclass2 and Plcass3


# In[42]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data = titanic);


# In[ ]:




