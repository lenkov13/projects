#!/usr/bin/env python
# coding: utf-8

# # Calculating the variables

# In[1]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
np.seterr(divide = 'ignore') 
import math

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'widget')

import ipywidgets
import pyreadstat

import ipywidgets as widgets

from plotnine import *
from plotnine.data import mpg

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import pairwise_distances

import plotly.express as px

import glob
import os


# opening of all df from the folder
# \  array(['ABBOTINDIA', 'bbRELIANCE', 'bbbbbbNTPC', 'bbbbbbbTCS'],
#       dtype=object)

# In[ ]:


# path_ord = r'E:\Projects\HFT\order' # use your path
# all_files_ord = glob.glob(os.path.join(path_ord , "*.sas7bdat"))

# path_trade = r'E:\Projects\HFT\trade' # use your path
# all_files_trade = glob.glob(os.path.join(path_trade , "*.sas7bdat"))

# def drop_q1_q2(x,q1 = 0.1,q2 = 0.9):
#     a = range(0,math.floor(x.index[-1]*q1))
#     b = range(math.floor(x.index[-1]*q2),x.index[-1])
#     x.drop(a,inplace = True)
#     x.drop(b,inplace = True)
#     x.reset_index(drop = True,inplace = True)
    
# df = pd.DataFrame()
# for i in all_files_ord:
#     temp, meta = pyreadstat.read_sas7bdat(i)
#     temp = temp.loc[temp.symbol == 'bbbbbbNTPC',:].reset_index(drop = True)
#     drop_q1_q2(temp)
#     df = pd.concat([df, temp], ignore_index=True)

# df.to_csv(r'E:\Projects\HFT\ntpc_orders.csv')


# In[3]:


TCSo = pd.read_csv(r'E:\Projects\HFT\tcs_orders.csv')
TCSt = pd.read_csv(r'E:\Projects\HFT\tcs_trades.csv')


# In[71]:


#pd.cut to create bins of 1 sec range
#1 sec = 65536
time_range = 65536
bins = math.floor((TCSo.quote_tm.iloc[-1] - TCSo.quote_tm.iloc[0]) / time_range)

ts, ts_lab = pd.cut(TCSo['quote_tm'], bins, retbins = True,labels = False)
TCSo['ts'] = ts
TCSo_gr = TCSo.groupby('ts')

ts_t, ts_lab_t = pd.cut(TCSt['trade_tm'], bins, retbins = True,labels = False)
TCSt['ts_t'] = ts_t
TCSt_gr = TCSt.groupby('ts_t')

c_ord = TCSo_gr.count().vol_orgnl
c_trade = TCSt_gr.count().trd_prc

#####Normalization of data in ts groups######
##### TCSo #####
# vol_tr = TCSo.groupby('ts').vol_orgnl.transform(
#     lambda x: (x - x.min()) / (x.max() - x.min())
# )
# prc_tr = TCSo.groupby('ts').limit_prc.transform(
#     lambda x: (x - x.min()) / (x.max() - x.min())
# )

# TCSo['vol_orgnl'] = vol_tr
# TCSo['limit_prc'] = prc_tr

# ##### TCSt #####
# trade_prc_tr = TCSt.groupby('ts_t').trd_prc.transform(
#     lambda x: (x - x.min()) / (x.max() - x.min()))
# TCSt['trd_prc'] = trade_prc_tr


# In[72]:


###Calculation of autocovariance for each interval###
#getting groups by bins and matching data
prc_gr = TCSo.groupby(['ts'],as_index = False).limit_prc        

dict_price = {key:list(TCSo.limit_prc.loc[prc_gr.groups[key]]) for key,value in prc_gr.groups.items()}
df_price = pd.DataFrame({key:pd.Series(value) for key, value in dict_price.items()})

log1 = np.log(df_price/df_price.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
log2 = np.log(df_price.shift(1)/df_price.shift(2)).replace([np.inf, -np.inf], np.nan).fillna(0)

auto_cov = ((log2 * log1).sum() / 
              df_price.count()).fillna(0)


# In[73]:


####number of ticks####
breadth = TCSo_gr.size()

####sum of volumes in group####
vol_all = TCSo_gr.vol_orgnl.sum().fillna(0)

####volume maximum in groupv
vol_max = TCSo_gr.vol_orgnl.max().fillna(0)

####average volume in group####
vol_avg = (vol_all/breadth).fillna(0)

####Sell/buy imbalance in order book####

#diff of sb 
#dif of means of B and S divided by sum of average 
imb = pd.DataFrame(TCSo.groupby(['buysell','ts'],as_index = False).vol_orgnl.sum())

imb_b = imb[imb.buysell == 'B']
imb_b = pd.merge(vol_all,imb_b, on = 'ts',how = 'left')

imb_s = imb[imb.buysell == 'S']
imb_s = pd.merge(vol_all,imb_s, on = 'ts',how = 'left')

imb = pd.merge(imb_b,imb_s, on = 'ts',how = 'left')

imb['imbalance'] = (imb.vol_orgnl_y_x - imb.vol_orgnl_y_y)/(imb.vol_orgnl_x_x/2) 
order_imbalance = imb['imbalance'].fillna(0)

#######Quoted spread########
#dif of means of B and S divided by sum of average 
spread = pd.DataFrame(TCSo.groupby(['buysell','ts'],as_index = False).limit_prc.mean().fillna(0))

spread_b = spread[spread.buysell == 'B']
spread_b = pd.merge(vol_all,spread_b, on = 'ts',how = 'left')
spread_b.set_index('ts', inplace=True)

spread_s = spread[spread.buysell == 'S']
spread_s = pd.merge(vol_all,spread_s, on = 'ts',how = 'left')
spread_s.set_index('ts', inplace=True)

spread = pd.merge(spread_b,spread_s, on = 'ts',how = 'left')

spread['dif'] = (spread.limit_prc_x - spread.limit_prc_y)
spread['avg'] = (spread.limit_prc_x + spread.limit_prc_y)/2
spread['spread'] = spread.dif/spread.avg

quoted_spread = spread['spread'].fillna(0)

#####order to trade ratio#####
count_orders = TCSo_gr.count().vol_orgnl
count_trade = TCSt_gr.count().trd_prc
order_to_trade = pd.merge(count_orders,count_trade, left_index=True, right_index=True,how = 'left')
order_to_trade = (order_to_trade.vol_orgnl / order_to_trade.trd_prc).fillna(0)

#####cancellation ration#####
dict_activity = {key:list(TCSo.activity_typ.loc[prc_gr.groups[key]]) for key,value in prc_gr.groups.items()}
df_activity = pd.DataFrame({key:pd.Series(value) for key, value in dict_activity.items()})
act_3 = df_activity[df_activity == 3].count()
act_3.name = 'act_3'
#for order book
canc_ratio_order = (act_3 / df_activity.count()).astype(
    float).fillna(0)

#for trade book
canc_ratio_trade = pd.merge(act_3, c_trade, left_index= True, right_index= True, how = 'left')
canc_ratio_trade = (canc_ratio_trade.act_3 / canc_ratio_trade.trd_prc).fillna(0).reset_index(drop = True)

#variance
variance = TCSo_gr.var().vol_orgnl

# #####HFT Ratio######

temp = TCSo.groupby(['ts','algo_ind','client_flg'],as_index = False).vol_orgnl.count()
temp = temp.loc[(temp.algo_ind == 0.0) & (temp.client_flg == 2.0)]
temp = temp.set_index(temp.ts, drop = True).vol_orgnl

hft_ratio = (temp/c_ord).fillna(0)


# In[74]:


list_var = [breadth,
            vol_all,
            vol_max,
            vol_avg, 
            order_imbalance, 
            auto_cov,
            quoted_spread, 
            order_to_trade, 
            canc_ratio_order, 
            canc_ratio_trade,
            hft_ratio,
            variance]

for i in list_var:
    i.reset_index(drop = True,inplace=True)


# In[79]:


# #####threshold for HFT#####
hft_copy = hft_ratio.copy()
# # threshold = 0.70

# # hft_copy[hft_copy >= threshold] = 1
# # hft_copy[(hft_copy < threshold)&(hft_copy != 1)]  
# hft_copy = hft_copy.reset_index(drop = True)
# # print('Percentage of HFT in bins: {}%'.format(hft_copy.sum()/len(hft_copy)*100))

# #new metric - HFT for the whole DF / HFT ratio in every bin
# HFT_df = len(TCSo[(TCSo.algo_ind == 0.0) & (TCSo.client_flg == 2.0)])/len(TCSo)
# HFT_bins = hft_copy.sum()/len(hft_copy)
# df_bins = HFT_df/HFT_bins
# print('HFT in df / HFT in bins: {}'.format(df_bins))

######## DATAFRAME WITH VARIABLES#######
df_var = pd.DataFrame({
'Breadth':breadth,
'Volumeall':vol_all,
'VolumeAvg':vol_avg,
# 'VolumeMax':vol_max,
'OrderImbalance':order_imbalance,
'AutoCov':auto_cov,
'QuotedSpread':quoted_spread,
'Cancellation to trade':canc_ratio_trade, 
'Cancellation to order':canc_ratio_order,
'Order-to-trade ratio':order_to_trade,
'Variance':variance
                      }).fillna(0)

def drop_q1_q2(x,q1 = 0.1,q2 = 0.9):
    a = range(0,math.floor(x.index[-1]*q1))
    b = range(math.floor(x.index[-1]*q2),x.index[-1])
    x.drop(a,inplace = True)
    x.drop(b,inplace = True)
    c = x.reset_index(drop = True)
    return c

df_var = drop_q1_q2(df_var)
df_var = df_var.replace(np.inf,0)

sc_X = StandardScaler()
var = sc_X.fit_transform(df_var)

hft_true = drop_q1_q2(hft_copy)

df_var = pd.DataFrame(var)

df_var['hft'] = hft_true

ls = pd.cut(hft_true, bins = np.arange(0,1.5,0.5), labels = [0,1],include_lowest=True)
df_var_ = np.array(df_var.drop('hft',axis = 1),dtype = 'int64')


# # Data split

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(
   df_var.drop('hft',axis = 1), df_var['hft'], test_size = 0.15, shuffle = True)


# # Dimensionality reduction

# ## PCA

# In[ ]:


from sklearn.decomposition import PCA


# In[82]:


pca = PCA(n_components=2)
pca_tr = pca.fit_transform(df_var_)


# In[83]:


grayscale_map = plt.get_cmap('rainbow')
plt.figure()
plt.scatter(x = pca_tr[:,0], 
            y = pca_tr[:,1],
            s = 1, 
            c = hft_true,
            cmap = grayscale_map)

plt.colorbar()


# # LLE

# In[40]:


from sklearn.manifold import LocallyLinearEmbedding


# In[64]:


lle = LocallyLinearEmbedding(n_components = 2, eigen_solver='auto')
lle_tr = lle.fit_transform(df_var_)


# In[65]:


plt.figure()
plt.scatter(x = lle_tr[:,0], 
            y = lle_tr[:,1],
            s = 1, 
            c = hft_true,
            cmap = grayscale_map)

plt.colorbar()


# # TSNE

# In[66]:


from sklearn.manifold import TSNE


# In[84]:


tsne = TSNE(n_components = 2, 
            n_jobs= 12,
            learning_rate='auto'
           )
tsne_tr = tsne.fit_transform(df_var_)


# In[85]:


plt.figure()
plt.scatter(x = tsne_tr[:,0], 
            y = tsne_tr[:,1],
            s = 1, 
            c = hft_true,
            cmap = grayscale_map)

plt.colorbar()


# # RF Optimization

# # Random Forest

# In[20]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[21]:


rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)


# In[139]:


rf_model.feature_importances_


# In[22]:


y_pred = rf_model.predict(X_test)


# In[23]:


print('MAE: ', mean_absolute_error(np.array(y_test), y_pred))
print('MSE: ', mean_squared_error(np.array(y_test), y_pred))


# In[78]:


importance = rf_model.feature_importances_
# importance.sort()


# Cross-validation

# In[ ]:


# important step - shuffling data
df_v = np.column_stack((df_var,df_hft_true))
np.random.shuffle(df_v)

X_train, X_test, y_train, y_test = train_test_split(
   df_v[:,0:9],df_v[:,10], test_size = 0.3, shuffle = True)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


# In[ ]:


# rf = RandomForestClassifier()

# kfold_validation = KFold(10)
# skfold_validation = StratifiedKFold(10)

results = cross_val_score(rf,df_v[:,0:9],df_v[:,10],cv = kfold_validation)
print(results)
print(np.std(results))
print(np.mean(results))


# Hyper parametres optimization

# In[ ]:


np.linspace(start = 100, stop = 1000, num = 10)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rs = RandomizedSearchCV(rf, 
                        param_dist, 
                        n_iter = 100, 
                        cv = 3, 
                        verbose = 1, 
                        n_jobs=-1, 
                        random_state=0)
rs.fit(X_train, y_train)
rs.best_params_
# {'n_estimators': 700,
# 'min_samples_split': 2,
# 'min_samples_leaf': 2,
# 'max_features': 'log2',
# 'max_depth': 11,
# 'bootstrap': True}


# In[ ]:


rf = RandomForestClassifier(n_estimators = 600,
                            min_samples_split = 23,
                            min_samples_leaf = 2,
                            max_features = 'sqrt',
                            max_depth = 15,
                            bootstrap = False,
                            verbose = 3)


# Important features: 1,2,8

# In[ ]:


y_test.shape


# In[ ]:


rf.fit(X_train,y_train)

print(f'accuracy of RF\n{rf.score(X_test,y_test)}\n')


# from the chart bellow we can see that features **possibly** could be separable

# In[ ]:


df_imp = pd.DataFrame(df_rf).loc[:,[1,2,8]]
df_imp['true'] = df_hft_true[0:1000]
sns.pairplot(df_imp,hue = 'true',height = 2)

