#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd

df = pd.read_csv(r"C:/Users/lucas.carillo/Downloads/Salary_dataset.csv")


# In[91]:


df.head()


# In[92]:


df.tail()


# In[93]:


df.info()


# In[94]:


df.drop(["Unnamed: 0"], axis =1, inplace = True)


# In[95]:


df.describe()


# In[96]:


df.isnull().sum()


# In[97]:


df.hist(figsize=(12,4))


# In[98]:


df.corr()


# In[99]:


df.duplicated().sum()


# In[100]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler

# Selecionando as colunas do DataFrame como uma matriz (DataFrame)
X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o transformador de coluna
transformer = make_column_transformer(
    (MinMaxScaler(), ['YearsExperience'])
)

# Ajustando o transformador de coluna aos dados de treinamento e transformando os dados
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.fit_transform(X_test)

# Imprimindo o resultado da transformação antes e depois do MinMaxScaler
print("Antes do MinMaxScaler:")
print(X_train.head())

print("Depois do MinMaxScaler:")
print(X_train_transformed[:5])


# In[101]:


X_train.shape


# In[102]:


y_train.shape


# In[103]:


type(X_train), type(y_train)


# In[104]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)


# In[105]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[106]:


import seaborn as sns
sns.regplot(x= y_test, y = y_pred)


# In[ ]:




