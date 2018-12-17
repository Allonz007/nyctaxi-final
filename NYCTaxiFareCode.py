
# coding: utf-8

# In[78]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("./train.csv", nrows = 1000000)


# In[79]:


import os # reading the input files we have access to
import matplotlib.pyplot as plt #data viz.
import seaborn as sb #data viz.

# Setting the random seed 
RSEED = 100


# In[80]:


train.head()


# In[81]:


# Saving the ID to submit
train_id = list(train.pop('key'))

train.describe()


# In[82]:


print(f"There are {len(train[train['fare_amount'] < 0])} negative fares.")
print(f"There are {len(train[train['fare_amount'] == 0])} $0 fares.")
print(f"There are {len(train[train['fare_amount'] > 100])} fares greater than $100.")


# In[83]:


# Remove na
train = train.dropna()
train.head()


# In[84]:


plt.figure(figsize = (10, 6))
sns.distplot(train['fare_amount']);
plt.title('Distribution of NYC Taxi Fares');


# In[85]:


train = train[train['fare_amount'].between(left = 2.5, right = 100)] #removal of fares <2.50 and >100 USD

# Bin the fare and convert to string
train['fare-bin'] = pd.cut(train['fare_amount'], bins = list(range(0, 50, 5))).astype(str)


# In[86]:


train[train.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');


# In[87]:


train['passenger_count'].value_counts().plot.bar(color = 'c', edgecolor = 'k');
plt.title('Passenger Counts'); plt.xlabel('Number of Passengers'); plt.ylabel('Count');


# In[88]:


train = train.loc[train['passenger_count'] < 6] #removal of passenger counts >6


# In[89]:


#how many data points are remaining from the 1M
print(f'Rows of data Remaining: {train.shape[0]}')


# In[90]:


# for a look at the latitude and longitude outliers
# this can be a place to also alter the modeling by changing the percentage +/-2.5%
for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']:
    print(f'{col.capitalize():17}: 2.5% = {round(np.percentile(train[col], 2.5), 2):5} \t 97.5% = {round(np.percentile(train[col], 97.5), 2)}')


# In[91]:


# Remove the 2.5% outliers in the latitude / longtiude.  This is an easy place to alter the size of the data by changing 
# the percentages of how much to lop off on the ends.
train = train.loc[train['pickup_latitude'].between(40, 42)]
train = train.loc[train['pickup_longitude'].between(-75, -72)]
train = train.loc[train['dropoff_latitude'].between(40, 42)]
train = train.loc[train['dropoff_longitude'].between(-75, -72)]

print(f'New number of observations: {train.shape[0]}')


# In[92]:


# Find the Absolute difference between pickup and dropoff - latitude and longitude
train['ab_diff_lat'] = (train['dropoff_latitude'] - train['pickup_latitude']).abs()
train['ab_diff_lon'] = (train['dropoff_longitude'] - train['pickup_longitude']).abs()


# In[93]:


# Finding how many taxi fares have the dropoff and pickup locations as the same coordinates
# 10,309 is a lot of taxi rides that didn't go anywhere but still had a fare over $2.50.  I'll keep this but
# think these could be an area for cleansing more

no_diff = train[(train['ab_diff_lat'] == 0) & (train['ab_diff_lon'] == 0)]
no_diff.shape


# In[94]:


# Let's dive into the modeling, shall we?
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

LR = LinearRegression()


# In[95]:


# Splitting the data between training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train, np.array(train['fare_amount']), 
                                                      stratify = train['fare-bin'],
                                                      random_state = RSEED, test_size = 900000)


# In[96]:


LR.fit(X_train[['ab_diff_lat', 'ab_diff_lon', 'passenger_count']], y_train)

print('Intercept', round(LR.intercept_, 4))
print('ab_diff_lat coef: ', round(LR.coef_[0], 4), 
      '\tab_diff_lon coef:', round(LR.coef_[1], 4),
      '\tpassenger_count coef:', round(LR.coef_[2], 4))


# In[118]:


from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)

def metrics(train_pred, test_pred, y_train, y_test):
    """Calculate metrics: Root mean squared error and mean absolute percentage error"""
    
    # Root-mean-squared error
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # Calculating absolute-percentage-error
    train_ap = abs((y_train - train_pred) / y_train)
    test_ap = abs((y_test - test_pred) / y_test)
    
    # Accounting for y values of zero
    train_ap[train_ap == np.inf] = 0
    train_ap[train_ap == -np.inf] = 0
    test_ap[test_ap == np.inf] = 0
    test_ap[test_ap == -np.inf] = 0
    
    train_map = 100 * np.mean(train_ap)
    test_map = 100 * np.mean(test_ap)
    
    return train_rmse, test_rmse, train_map, test_map

def evaluate(model, features, X_train, X_test, y_train, y_test):
    """Mean absolute percentage error"""
    
    # Make predictions
    train_pred = model.predict(X_train[features])
    test_pred = model.predict(X_test[features])
    
    # Get metrics
    train_rmse, test_rmse, train_map, test_map = metrics(train_pred, test_pred,
                                                             y_train, y_test)
    
    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_map, 2)}')
    print(f'Testing:    rmse = {round(test_rmse, 2)} \t mape = {round(test_map, 2)}')


# In[119]:


evaluate(LR, ['ab_diff_lat', 'ab_diff_lon', 'passenger_count'], 
        X_train, X_test, y_train, y_test)


# In[99]:


# Now lets get a Baseline to compare with all of this data
train_mean = y_train.mean()

# Create list of the same prediction for every observation
train_pred = [train_mean for _ in range(len(y_train))]
test_pred =  [train_mean for _ in range(len(y_test))]

tnr, ttr, tnm, ttm = metrics(train_pred, test_pred, y_train, y_test)

print(f'Baseline Training:   rmse = {round(tnr, 2)} \t mape = {round(tnm, 2)}')
print(f'Baseline Testing:    rmse = {round(ttr, 2)} \t mape = {round(ttm, 2)}')


# In[110]:


sns.distplot(train['fare_amount'])
plt.title('Dist. of Linear Regression Pred.');


# In[114]:


from sklearn.ensemble import RandomForestRegressor

# Create the random forest
random_forest = RandomForestRegressor(n_estimators = 20, max_depth = 20, 
                                      max_features = None, oob_score = True, 
                                      bootstrap = True, verbose = 1, n_jobs = -1)

# Train on data
random_forest.fit(X_train[[ 'ab_diff_lat', 'ab_diff_lon', 'passenger_count']], y_train)


# In[115]:


evaluate(random_forest, ['ab_diff_lat', 'ab_diff_lon', 'passenger_count'],
         X_train, X_test, y_train, y_test)

