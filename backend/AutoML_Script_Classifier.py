#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# In[4]:

def runtool(file_url,target):
    print("Running Tool")
    dataset = pd.read_csv(file_url)
    target_column = target
    dataset
    
    
    # In[5]:
    
    
    dataset.info()
    if "id" or "ID" in dataset.columns:
        if("id" in dataset.columns):
            dataset.drop("id",inplace=True,axis=1)
        elif("ID" in dataset.columns):
            dataset.drop("ID",inplace=True,axis=1)
    
    
    # In[6]:
    
    
    datatypes = dataset.dtypes
    datatypes
    
    
    # In[7]:
    
    
    cat_cols = []
    num_cols = []
    id1 = "id"
    
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
    
    
    # In[8]:
    
    
    cat_cols
    
    
    # In[9]:
    
    
    cat_data = []
    num_data = []
    if(len(cat_cols)>0):
        cat_data = dataset[cat_cols]
    if(len(num_cols)>0):
        num_data = dataset[num_cols]
    y_data = dataset[target_column]
    
    
    # In[10]:
    
    
    num_data,cat_data
    
    
    # In[11]:
    
    
    if len(num_data):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
        imputer.fit(num_data.iloc[:,:].values)
        num_data = pd.DataFrame(imputer.transform(num_data.iloc[:,:].values))
        num_data.columns = num_cols
    num_data
    
    
    # In[12]:
    
    
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
    
    
    # In[13]:
    
    
    cat_data = df[cat_cols]
    num_data = df[num_cols]
    y_data = df[target_column]
    df.drop(target_column,inplace=True,axis=1)
    len(y_data)
    
    
    # In[14]:
    
    
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
    
    
    # In[15]:
    
    
    OH_X
    
    
    # In[16]:
    
    
    count = y_data.value_counts()
    print(count)
    print(max(count))
    print(min(count))
    cond = min(count)/max(count)
    print(cond)
    
    
    # In[17]:
    
    
    from imblearn.over_sampling import ADASYN
    from collections import Counter
    if(cond<0.3):
        counter = Counter(y_data)
        print('before :',counter)
        ADA = ADASYN(random_state=130,sampling_strategy='minority')
        OH_X,y_data = ADA.fit_resample(OH_X,y_data)
        counter = Counter(y_data)
        print("after :",counter)
    
    
    # In[18]:
    
    
    from sklearn.preprocessing import StandardScaler
    
    
    # In[19]:
    
    
    sc_x = StandardScaler()
    
    
    # In[20]:
    
    
    OH_X = sc_x.fit_transform(OH_X)
    
    
    # In[21]:
    
    
    OH_X
    
    
    # In[ ]:
    
    
    
    
    
    # In[22]:
    
    
    from sklearn.model_selection import train_test_split
    
    
    # In[23]:
    
    
    X_train,X_test,y_train,y_test = train_test_split(OH_X,y_data,test_size=0.2,random_state=0)
    
    
    # In[24]:
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix
    nodes = [10,15,20,25,50,100,200]
    accuracy =[]
    for node in nodes:    
        model = RandomForestClassifier(criterion = 'entropy', random_state=0, n_estimators=node)
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        accs = accuracy_score(y_test,preds)
        accuracy.append(accs)
        print(confusion_matrix(y_test,preds),'No of Estimators: ', node,accs)
    
    sns.lineplot(x=nodes,y=accuracy)
    
    
    # In[25]:
    
    
    from xgboost import XGBClassifier   #XGBoostClassifier
    classifier = XGBClassifier(max_depth=6)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    # In[26]:
    
    
    # rfc=RandomForestClassifier(random_state=42)
    
    
    # In[27]:
    
    
    # param_grid = { 
    #     'n_estimators': [200, 500],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth' : [4,5,6,7,8],
    #     'criterion' :['gini', 'entropy']
    # }
    
    
    # In[30]:
    
    
    # from sklearn.model_selection import GridSearchCV
    # CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    # CV_rfc.fit(X_train, y_train)
    # CV_rfc.best_params_
    
    
    # In[31]:
    
    
    # rfc=RandomForestClassifier(random_state=42,criterion= 'gini',max_depth=4,max_features= 'auto',n_estimators= 200)
    
    
    # In[33]:
    
    
    # rfc.fit(X_train, y_train)
    # preds = rfc.predict(X_test)
    # accs = accuracy_score(y_test,preds)
    # accuracy.append(accs)
    # print(confusion_matrix(y_test,preds),'Accuracy: ',accs)
    
    return accuracy
    # In[ ]:
    
    
    
    
