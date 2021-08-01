#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score


# ### Loading Dataset

# In[6]:


def runtool(file_url,target):
    print("Running Tool")
    dataset = pd.read_csv(file_url)
    dataset
    return target

# In[7]:


if "Unnamed: 0" in dataset.columns:
    dataset.drop("Unnamed: 0",axis=1,inplace=True)
dataset


# ## Preprocessing

# ### Remove Date Columns

# In[8]:


if "date" in dataset.columns:
    dataset.drop("date",inplace=True,axis=1)
    print("date Deleted")
if "Date" in dataset.columns:
    dataset.drop("Date",inplace=True,axis=1)
    print("Date Deleted")


# ### Remove Dollar signs

# In[9]:


dollar = '$'
prices_list = []
if(len(dataset) > 100):
    for column in dataset.columns:
        X = dataset[column][:50].values
        for val in X:
            if( "$" in str(val)):
                prices_list.append(column)
                break;
            break;
            
prices_list


# In[10]:


new_2darray = []
if(prices_list):
    cost_values = dataset[prices_list].values
    for array in cost_values:
        new_array = []
        for val in array:
            new_val = (re.sub("[^0-9]", "", val))
            if new_val:
                new_val = float(new_val)
            else:
                new_val = np.NaN
            new_array.append(new_val)
        new_2darray.append(new_array)
new_2darray


# In[11]:


if(prices_list):
    prices_data = pd.DataFrame(new_2darray,columns= prices_list)


# In[12]:


if(prices_list):
    dataset.drop(prices_list,inplace=True,axis=1)


# In[13]:


if(prices_list):
    dataset[prices_list] = prices_data


# ### Define Index / ID Columns and Target Columns

# In[14]:


target_column = 'DEATH_EVENT'
id1 = "id"


# ### Categorize into numerical and categorical

# In[15]:


dataset.info()


# In[16]:


datatypes = dataset.dtypes
datatypes


# In[17]:


cat_cols = []
num_cols = []

columns = dataset.columns
if id1 in dataset.columns:
    dataset.drop(id1,inplace=True,axis=1)

for i in range(len(datatypes)):
    if datatypes.index[i] != target_column:
        if datatypes[i]=='object':
            unqval = dataset[datatypes.index[i]].nunique()
            if (unqval < 30):
                cat_cols.append(datatypes.index[i])
            else:
                del dataset[datatypes.index[i]]
                print('Deleted: ',datatypes.index[i])
        else:
            num_cols.append(datatypes.index[i])
num_cols


# In[18]:


cat_cols


# In[19]:


cat_data = []
num_data = []
if(len(cat_cols)>0):
    cat_data = dataset[cat_cols]
if(len(num_cols)>0):
    num_data = dataset[num_cols]
y_data = dataset[target_column]


# In[20]:


num_data,cat_data


# ### Outlier Detection 

# In[21]:


def outlier_detect(df):
    if (len(num_data)>0):        
        for i in df.describe().columns:
            Q1=df.describe().at['25%',i]
            Q3=df.describe().at['75%',i]
            IQR=Q3 - Q1
            LB=Q1 - 1.5 * IQR
            UB=Q3 + 1.5 * IQR
            x=np.array(df[i])
            p=[]
            for j in x:
                if j < LB or j>UB:
                   p.append(j)
            print('\n Outliers for Column : ', i, ' Outliers count ', len(p))
            print(p)
    else:
        print("Num_data is empty")

        
outlier_detect(num_data)


# ### Treat Missing Values

# In[22]:


if len(num_data):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
    imputer.fit(num_data.iloc[:,:].values)
    num_data = pd.DataFrame(imputer.transform(num_data.iloc[:,:].values))
    num_data.columns = num_cols
num_data


# In[23]:


if(len(cat_data)>0 and len(num_data)>0):
    print("1")
    df = pd.concat([num_data,cat_data,y_data],axis=1)
    df = df.dropna().reset_index(drop=True)
    df.isnull().sum()
elif(len(num_data) > 0):
    print("2")
    df = pd.concat([num_data,y_data],axis=1)
    df = df.dropna().reset_index(drop=True)
else:
    print("3")
    df = pd.concat([cat_data,y_data],axis=1)
    df = df.dropna().reset_index(drop=True)
(df)


# In[24]:


cat_data = df[cat_cols]
num_data = df[num_cols]
y_data = df[target_column]
df.drop(target_column,inplace=True,axis=1)
len(y_data)


# ### One-hot encoding the categorical data

# In[25]:


#if len(cat_cols):
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = pd.DataFrame(OH_encoder.fit_transform(df[cat_cols]))

# One-hot encoding removed index; put it back
X_encoded.index = df.index

# Remove categorical columns (will replace with one-hot encoding)
num_X = df.drop(cat_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X = pd.concat([num_X, X_encoded], axis=1)


# In[26]:


OH_X


# ### Sampling/ Unbalanced Check

# In[27]:


count = y_data.value_counts()
print(count)
print(max(count))
print(min(count))
cond = min(count)/max(count)
print(cond)


# In[28]:


from imblearn.over_sampling import ADASYN
from collections import Counter
if(cond<0.3):
    counter = Counter(y_data)
    print('before :',counter)
    ADA = ADASYN(random_state=130,sampling_strategy='minority')
    OH_X,y_data = ADA.fit_resample(OH_X,y_data)
    counter = Counter(y_data)
    print("after :",counter)


# ### Scaling

# In[29]:


from sklearn.preprocessing import StandardScaler


# In[30]:


sc_x = StandardScaler()


# In[31]:


OH_X = sc_x.fit_transform(OH_X)


# In[32]:


OH_X


# In[33]:


model_accuracies = {}


# In[34]:


from sklearn.metrics import roc_auc_score,accuracy_score, classification_report
def write_report(model, remarks, model_name):
 preds = model.predict(X_test)
 report = classification_report(y_test,preds)
 balanced_accuracy = '-'
 roc_score = roc_auc_score(y_test, preds)
 accuracyscore = accuracy_score(y_test, preds)
 print(f'\n{model}\n {report}\n  accuracy score: {accuracyscore}\n Roc auc score: {roc_score}\n Remarks: {remarks}')
 with open(".//reports//"+model_name+"_report.txt", "w") as myfile:
   myfile.write(f'\n{model}\n {report}\n accuracy score: {accuracyscore}\n Roc auc score: {roc_score}\n Remarks: {remarks}')


# ### Splitting Dataset

# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train,X_test,y_train,y_test = train_test_split(OH_X,y_data,test_size=0.2,random_state=0)


# ## MODEL TRAINING

# ### Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression(random_state = 0)
LR_classifier.fit(X_train, y_train)

y_pred_lr = LR_classifier.predict(X_test)


# In[38]:


cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
model_accuracies["LogisticRegression"] = lr_accuracy
print(lr_accuracy)


# In[45]:


write_report(LR_classifier,"good Model","Logistic Regression")


# ### Decision Tree Classifier

# In[46]:


from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(X_train, y_train)

y_pred_dr = DT_classifier.predict(X_test)


# In[47]:


cm = confusion_matrix(y_test, y_pred_dr)
print(cm)
dt_accuracy = accuracy_score(y_test, y_pred_dr)
model_accuracies["DecisionTreeClassifer"] = dt_accuracy
print(dt_accuracy)


# In[48]:


write_report(DT_classifier,"good Model","Decision Tree Classifer")


# ### Random Forest classifier

# In[49]:


from sklearn.ensemble import RandomForestClassifier
nodes = [10,15,20,25,50,100,200]
accuracy =[]
for node in nodes:    
    RandomForestclassifier = RandomForestClassifier(criterion = 'entropy', random_state=0, n_estimators=node)
    RandomForestclassifier.fit(X_train,y_train)
    preds = RandomForestclassifier.predict(X_test)
    accs = accuracy_score(y_test,preds)
    accuracy.append(accs)
    print(confusion_matrix(y_test,preds),'No of Estimators: ', node,accs)

# sns.lineplot(x=nodes,y=accuracy)
model_accuracies["RandomForestclassifier"] = accs


# In[50]:


write_report(RandomForestclassifier,"good Model","RandomForestClassifier")


# ### XgBoost Classifier

# In[51]:


from xgboost import XGBClassifier   #XGBoostClassifier
XGboostclassifier = XGBClassifier(max_depth=6)
XGboostclassifier.fit(X_train, y_train)
y_pred = XGboostclassifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
xgboost_accuracy = accuracy_score(y_test, y_pred)
model_accuracies['XGboostclassifier'] = xgboost_accuracy


# In[52]:


write_report(XGboostclassifier,"good Model","XGboostclassifier")


# ### SVM Classifier
# 

# In[53]:


from sklearn.svm import SVC
SVCclassifier = SVC(kernel="rbf", random_state=0)
SVCclassifier.fit(X_train, y_train)
y_pred_svm = SVCclassifier.predict(X_test)


# In[54]:


cm = confusion_matrix(y_test, y_pred_svm)
print(cm)
svc_accuracy = accuracy_score(y_test, y_pred_svm)
model_accuracies["SVCclassifier"] = svc_accuracy


# In[55]:


write_report(SVCclassifier,"good Model","SVCclassifier")


# ### Naive Bayes 

# In[56]:


from sklearn.naive_bayes import GaussianNB
GaussianNBclassifier = GaussianNB()
GaussianNBclassifier.fit(X_train, y_train)
y_pred_nb = GaussianNBclassifier.predict(X_test)


# In[59]:


cm = confusion_matrix(y_test, y_pred_nb)
print(cm)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
model_accuracies["GaussianNBclassifier"] = nb_accuracy


# In[60]:


write_report(GaussianNBclassifier,"good Model","GaussianNBclassifier")


# ### Ada Boost

# In[61]:


from sklearn.ensemble import AdaBoostClassifier
AdaboostClassifier = AdaBoostClassifier(n_estimators=100,learning_rate=1)
model = AdaboostClassifier.fit(X_train, y_train)
y_pred_ada = AdaboostClassifier.predict(X_test)


# In[62]:


cm = confusion_matrix(y_test, y_pred_ada)
print(cm)
adaboost_accuracy = accuracy_score(y_test, y_pred_ada)
model_accuracies["AdaboostClassifier"] = adaboost_accuracy


# In[63]:


write_report(AdaboostClassifier,"Good report","AdaboostClassifier")


# ### Finding the best model

# In[64]:


bestmodel = max(model_accuracies, key = model_accuracies.get)
print(bestmodel)
print(model_accuracies[bestmodel])


# In[65]:


import pickle
# open a file, where you ant to store the data
file = open('classification.pkl', 'wb')

# dump information to that file
pickle.dump(RandomForestclassifier, file)


# In[ ]:




