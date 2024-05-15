#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# make a dataframe, and loanstatenew should be used as index for it

data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


print(data_info.loc['revol_util']['Description'])


# In[4]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[5]:


feat_info('mort_acc')


# In[6]:


df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[7]:


df.info()


# In[8]:


sns.countplot(x='loan_status',data=df)


# In[9]:


plt.figure(figsize=(14,7))
sns.displot(df['loan_amnt'],kde=False,bins=70)


# In[10]:


df.corr(numeric_only=True)


# In[11]:


plt.figure(figsize=(13,7))
sns.heatmap(df.corr(numeric_only=True), annot=True,cmap='viridis')
plt.ylim=(10,0)


# In[12]:


feat_info('installment')


# In[13]:


feat_info('loan_amnt')


# In[14]:


sns.scatterplot(x='installment',y='loan_amnt',data=df)


# In[15]:


sns.boxplot(x='loan_status', y='loan_amnt',data=df)


# In[16]:


df.groupby('loan_status')['loan_amnt'].describe()


# In[17]:


df['grade'].unique()


# In[18]:


df['sub_grade'].unique()


# In[19]:


feat_info('sub_grade')


# In[20]:


grade_order = sorted(df['grade'].unique())
sns.countplot(x='grade',data=df,hue='loan_status',order=grade_order)


# In[21]:


plt.figure(figsize=(20,7))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,hue='loan_status',order=subgrade_order)


# In[22]:


plt.figure(figsize=(20,7))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm')


# In[23]:


f_g = df[ (df['grade'] == 'G') | (df['grade']=='F')]
plt.figure(figsize=(20,7))
subgrade_order = sorted(f_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_g,order=subgrade_order,palette='coolwarm')


# In[24]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0 })


# In[25]:


df[['loan_repaid','loan_status']]


# In[28]:


df.corr(numeric_only=True)['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# In[29]:


df.head()


# In[30]:


len(df)


# In[31]:


df.isnull().sum()


# In[32]:


100 * df.isnull().sum() / len(df)


# In[33]:


feat_info('emp_title')


# In[34]:


feat_info('emp_length')


# In[35]:


df['emp_title'].unique()


# In[36]:


df['emp_title'].nunique()


# In[37]:


df['emp_title'].value_counts()


# In[38]:


df = df.drop('emp_title',axis=1)


# In[39]:


sorted(df['emp_length'].dropna().unique())


# In[40]:


emp_length_order = [ '< 1 year',
                    '1 year',
   
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years']


# In[41]:


plt.figure(figsize=(14,6))
sns.countplot(data=df,x='emp_length',order=emp_length_order)


# In[42]:


plt.figure(figsize=(14,6))
sns.countplot(data=df,x='emp_length',order=emp_length_order,hue='loan_status')


# In[43]:


emp_chargedoff = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
emp_chargedoff


# In[44]:


emp_paidoff = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
emp_paidoff


# In[45]:


emp_len = emp_chargedoff/(emp_chargedoff+emp_paidoff)
emp_len


# In[46]:


emp_len.plot(kind='bar')


# In[47]:


df = df.drop('emp_length',axis=1)


# In[48]:


df.isnull().sum()


# In[49]:


df['purpose'].head()


# In[50]:


feat_info('purpose')


# In[51]:


df['title'].head()


# In[52]:


feat_info('title')


# In[53]:


df = df.drop('title',axis=1)


# In[54]:


feat_info('mort_acc')


# In[55]:


df['mort_acc'].value_counts()


# In[56]:


df.corr(numeric_only=True)['mort_acc'].sort_values()


# In[57]:


df.groupby('total_acc').mean(numeric_only=True)


# In[58]:


total_acc_avg = df.groupby('total_acc').mean(numeric_only=True)['mort_acc']


# In[59]:


total_acc_avg


# In[60]:


def fill_mort_acc(total_acc,mort_acc):
    
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[61]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)


# In[62]:


df.isnull().sum()


# In[63]:


df= df.dropna()


# In[64]:


df.isnull().sum()


# In[65]:


df.select_dtypes(['object']).columns


# In[66]:


feat_info('term')


# In[67]:


df['term'].value_counts()


# In[68]:


df['term'] = df['term'].apply(lambda term: int(term[:3])) 


# In[69]:


df['term'].value_counts()


# In[70]:


df = df.drop('grade',axis=1)


# In[71]:


dummies = pd.get_dummies(df['sub_grade'],drop_first=True)

df = pd.concat([df.drop('sub_grade',axis=1),dummies],axis=1)


# In[72]:


df.columns


# In[75]:


dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)

df = pd.concat([df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1),dummies],axis=1)


# In[76]:


df['home_ownership'].value_counts()


# In[78]:


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')


# In[79]:


dummies = pd.get_dummies(df['home_ownership'],drop_first=True)

df = pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)


# In[80]:


df['address']


# In[84]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# In[85]:


df['zip_code'].value_counts()


# In[86]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)

df = pd.concat([df.drop('zip_code',axis=1),dummies],axis=1)


# In[87]:


df = df.drop('address',axis=1)


# In[88]:


feat_info('issue_d')


# In[89]:


df = df.drop('issue_d',axis=1)


# In[90]:


feat_info('earliest_cr_line')


# In[92]:


df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))


# In[93]:


df['earliest_cr_line']


# In[94]:


df['earliest_cr_line'].value_counts()


# In[95]:


from sklearn.model_selection import train_test_split


# In[96]:


df = df.drop('loan_status',axis=1)


# In[98]:


X = df.drop('loan_repaid',axis=1).values


# In[99]:


y = df['loan_repaid'].values


# In[100]:


df = df.sample(frac=0.1,random_state=101)
print(len(df))


# In[101]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[102]:


from sklearn.preprocessing import MinMaxScaler


# In[103]:


scaler = MinMaxScaler()


# In[104]:


X_train = scaler.fit_transform(X_train)


# In[105]:


X_test = scaler.transform(X_test)


# In[106]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[108]:


X_train.shape


# In[109]:


model = Sequential()

model.add(Dense(78,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[110]:


model.fit(x=X_train,y=y_train,epochs=25,batch_size=256,validation_data=(X_test,y_test))


# In[111]:


from tensorflow.keras.models import load_model


# In[112]:


model.save('lendingClubmodel.h5')


# In[113]:


losses = pd.DataFrame(model.history.history)


# In[114]:


losses


# In[115]:


losses.plot()


# In[116]:


from sklearn.metrics import classification_report,confusion_matrix


# In[118]:


predictions = model.predict(X_test)

predictions = np.round(predictions).astype(int)


# In[121]:


print(classification_report(y_test,predictions))


# In[122]:


df['loan_repaid'].value_counts()


# In[123]:


31664/len(df)


# In[124]:


confusion_matrix(y_test,predictions)


# In[125]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[129]:


new_customer = scaler.transform(new_customer.values.reshape(1,78))


# In[130]:


new_customer


# In[131]:


model.predict(new_customer)


# In[133]:


df.iloc[random_ind]['loan_repaid']


# In[ ]:




