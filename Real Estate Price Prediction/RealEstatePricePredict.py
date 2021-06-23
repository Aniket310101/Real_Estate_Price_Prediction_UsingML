#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[6]:


import pandas as pd


# In[7]:


df1 = pd.read_csv("bengaluru_House_Data.csv")
df1.head()


# In[10]:


df1.shape


# # Data Cleansing

# In[9]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# In[11]:


df2.isnull().sum()


# In[12]:


df3 = df2.dropna()
df3.isnull().sum()


# In[13]:


df3.shape


# In[14]:


df3['size'].unique()


# In[15]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[16]:


df3.head()


# In[17]:


df3['bhk'].unique()


# In[18]:


df3[df3.bhk>20]


# In[19]:


df3.total_sqft.unique()


# In[20]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[21]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[22]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[23]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head(3)


# In[24]:


df4.loc[30]


# In[25]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[26]:


len(df5.location.unique())


# In[27]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[28]:


len(location_stats[location_stats<=10])


# In[29]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[30]:


len(df5.location.unique())


# In[31]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[32]:


df5.head(10)


# # OUTLIER REMOVAL

# In[33]:


df5[df5.total_sqft/df5.bhk < 300].head()


# Remove all the row with bedroom size < 300 sqft.

# In[34]:


df5.shape


# In[35]:


df6 = df5[~ (df5.total_sqft/df5.bhk < 300)]
df6.shape


# In[36]:


df6.price_per_sqft.describe()


# Remove all the House records with extremely high or low price.

# In[37]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
df7.shape


# In[38]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location)&(df.bhk==2)]
    bhk3 = df[(df.location==location)&(df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='blue',label='3 BHK',s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# Remove records for all those houses whose House price for 2 BHK is more than 3 BHK for the same square feet area.

# In[39]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
            
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)
df8.shape


# In[40]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[41]:


df8.bath.unique()


# In[42]:


df8[df8.bath>10]


# In[44]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# Anytime if the number of Bathrooms is greater than the (number of bedrooms+ 2) mark it as an outlier. 

# In[45]:


df8[df8.bath>df8.bhk+2]


# In[46]:


df9 = df8[df8.bath<df8.bhk + 2]
df9.shape


# NOW WE DROP size AND price_per_sqft FEATURES

# In[47]:


df10 = df9.drop(['size','price_per_sqft'], axis='columns')
df10.head(3)


# # MODEL BUILDING

# ML model cannot interpret Text Data. So, we need to covert location into a numeric value using DUMMIES (hot encode).

# In[94]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[95]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head(3)


# In[96]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[97]:


df12.shape


# In[98]:


x = df12.drop('price',axis='columns')
x.head()


# In[99]:


y = df12.price
y.head()


# x contains all independent features
# y contains all the dependent features (price)

# We divide datasets into train and test datatset

# In[100]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10) 


# In[101]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)


# Score is 84% (How good our model is).

# K-fold Cross validation (Which Regressiin Technique is best suitable)

# In[102]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)


# Now we check for some other regression techniques to find the model with best score. We use GrudSearchCV method provided by sklearn which runs our model in different parameters and give the best score and the best model.

# In[103]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random','cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best','random']
            }
        }
        
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
        
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)


# Linear Regression is the best model.

# In[104]:


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns==location)[0][0]
    
    Xp = np.zeros(len(x.columns))
    Xp[0] = sqft
    Xp[1] = bath
    Xp[2] = bhk
    if loc_index >= 0:
        Xp[loc_index] = 1
        
    return lr_clf.predict([Xp])[0]


# In[105]:


x.columns


# In[106]:


np.where(x.columns=='2nd Phase Judicial Layout')[0][0]


# In[113]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[112]:


predict_price('Indira Nagar',1000, 3, 3)


# Now that the model is ready, we need to import the file using pickle.

# In[110]:


import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)


# Import JSon File.

# In[111]:


import json
columns = {
    'data_columns': [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

