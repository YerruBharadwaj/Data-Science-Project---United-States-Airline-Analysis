#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from scipy.stats import randint
from sklearn import metrics


# In[2]:


Airlines = pd.read_excel('C:\\Users\\Lenovo\\Desktop\\airline analysis Datascience Project\\Capstone_3\\Airlines.xlsx')


# In[3]:


Runways = pd.read_excel('C:\\Users\\Lenovo\Desktop\\airline analysis Datascience Project\\Capstone_3\\runways.xlsx')


# In[4]:


Airlines.head()


# In[5]:


Runways.head(5)


# In[6]:


Runway1 = Runways.drop(['length_ft', 'width_ft', 'le_ident', 'airport_ident', 'airport_ref', 'surface', 'le_latitude_deg', 'le_longitude_deg', 'le_elevation_ft', 'le_heading_degT', 'le_displaced_threshold_ft', 'he_ident', 'he_latitude_deg', 'he_longitude_deg', 'he_elevation_ft', 'he_heading_degT', 'he_displaced_threshold_ft'], axis=1)


# In[7]:


Runway1.isnull().sum()


# In[8]:


Airlines.isnull().sum()


# In[9]:


df = pd.merge(Airlines,Runway1,on ="id",how = "outer")


# In[10]:


# Delete id column
df = df.drop(columns = ['id'],axis =1)


# 0.0.1 Data Analysis

# In[11]:


# Bring only the nuemric columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_column_names = df.select_dtypes(include=numerics).columns
print(numeric_column_names, '\n')

# Bring only the object columns (strings)
objects = 'object'
object_column_names = df.select_dtypes(include=objects).columns
print(object_column_names, '\n')


# In[12]:


df.isnull().sum()


# In[13]:


df = df.dropna().reset_index(drop = True)


# In[14]:


df.isnull().sum()


# In[15]:


df.info()


# In[16]:


df.nunique()


# In[17]:


df.info()


# In[18]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_column_names = df.select_dtypes(include=numerics).columns
print(numeric_column_names, '\n')

# Bring only the object columns (strings)
objects = 'object'
object_column_names = df.select_dtypes(include=object).columns
print(object_column_names, '\n')


# In[19]:


import seaborn as sns


# In[20]:


for column in object_column_names:
    value_counts_column = df[column].value_counts()
    if value_counts_column.shape[0] > 20:
        value_counts_column = value_counts_column.sort_values(ascending = False)
        value_counts_column = value_counts_column[:20]
    ax = sns.barplot(y=value_counts_column.values, x=value_counts_column.index)
    ax = ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    plt.show()


# In[21]:


for column in object_column_names:
    value_counts_column = df[column].value_counts()
    if value_counts_column.shape[0] > 20:
        value_counts_column = value_counts_column.sort_values(ascending = False)
        value_counts_column = value_counts_column[:20]
        df_aux = df[df[column].isin(value_counts_column[:20].index)]
        ax = sns.histplot(data=df_aux, x=column, hue='Delay', multiple="dodge", shrink=.8)
        plt.xticks(rotation=45)
        plt.show()
    else:
        ax = sns.histplot(data=df, x=column, hue='Delay', multiple="dodge", shrink=.8)
        plt.show()


# In[22]:


for column in numeric_column_names:
    ax = sns.histplot(data=df, x=column, hue='Delay', kde=True)
    plt.show()


# In[23]:


df.corr = pd.read_excel('C:\\Users\\Lenovo\\Desktop\\airline analysis Datascience Project\\Capstone_3\\Data Dictionary.xlsx')


# In[24]:


df.corr()


# In[25]:


# Correlation matrix graph of the data set
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df.corr(), annot=True, fmt=".3f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# In[26]:


feature=df.drop(columns=['Delay','AirportTo', 'AirportFrom'])
target = df['Delay']


# 0.0.2 Classification
# 
# Here we will try to process the categorical columns with the function OrdinalEncoder that just
# creates a value for each category in the columns, so we don’t change the number of columns, just
# transform them in a number, so the model can process.

# In[27]:


# Divide in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature , target,
                                                    shuffle = True,
                                                    test_size=0.2,
                                                    random_state=1)


# In[28]:


from sklearn.preprocessing import OrdinalEncoder

# Function to create the new columns with Oridinal Encoding


# In[29]:


def ordinal_encoding(X_train, X_test, columns):
    
    # Create the encoder
    ord_enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=-1)
    ord_enc.fit(X_train[columns])
    
    # Transfrom both train and test datasets (separately)
    new_columns_train = ord_enc.transform(X_train[columns])
    new_columns_test = ord_enc.transform(X_test[columns])
    
    # Creating the name of the columns, being the same name as the original but with a _num at the end
    column_num_names = columns + '_num'
    
    # Creating a dataframe for the new columns
    new_columns_train = pd.DataFrame(new_columns_train, columns = column_num_names, index = X_train.index)
    new_columns_test = pd.DataFrame(new_columns_test, columns = column_num_names, index = X_test.index)
    
    # Concating the new columns to the original datasets
    X_train = pd.concat([X_train, new_columns_train], axis=1)
    X_test = pd.concat([X_test, new_columns_test], axis=1)
    
    return X_train, X_test


# In[30]:


# Delete the columns AirportTo and AirportFrom, because we're going to use only the columns Airline for Categorical column
object_column_names = object_column_names.drop(['AirportTo', 'AirportFrom'],errors = 'ignore')
X_train, X_test = ordinal_encoding(X_train, X_test, object_column_names)

# Drop the columns we will not use
X_train = X_train.drop(columns = object_column_names)
X_test = X_test.drop(columns = object_column_names)


# In[31]:


# Function to evaluate easily multiple models
def evaluate_model(model, x_test, y_test):
    
    y_pred = model.predict(x_test)
    
    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'cm': cm}

# Show results from evaluate_model
def print_metrics(eval):
    # Print result
    print('Accuracy:', (eval['acc'] * 100).round(2), '%')
    print('Precision:', (eval['prec']* 100).round(2), '%')
    print('Recall:', (eval['rec']* 100).round(2), '%')
    print('F1 Score:', (eval['f1'] * 100).round(2), '%')
    print('Cohens Kappa Score:', (eval['kappa'] * 100).round(2), '%')
    print('Confusion Matrix:\n', eval['cm'])


# In[32]:


# Show results from cross_validation()
def print_results(results):
    mean = results['test_score'].mean()
    std = results['test_score'].std()
    print('Accuracy mean %.2f' % (mean * 100))
    print('Accuracy interval %.2f %.2f' % ((mean - 2 * std) * 100, (mean + 2*std) * 100))


# In[33]:


from sklearn.preprocessing import StandardScaler

# Function to scale the dataset
def scaling(X_train, X_test):
    scaler_feature = StandardScaler()
    scaler_feature.fit(X_train)
    
    X_train_scaled = scaler_feature.transform(X_train)
    X_test_scaled = scaler_feature.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scaling(X_train, X_test)


# In[34]:


# Function to train and test the models and show the results in a organized way
def run_classification(model, X_train, X_test, y_train, y_test):
    cv = KFold(n_splits=5, shuffle=True)
    results = cross_validate(model, X_train, y_train, cv = cv)
    
    print_results(results)
    print('------------------')
    
    # Evaluate Model
    model.fit(X_train, y_train)
    dtc_eval = evaluate_model(model, X_test, y_test)
    print_metrics(dtc_eval)


# 0.0.3 Dummy (Baseline)

# In[35]:


from sklearn.dummy import DummyClassifier

dummy_stratified = DummyClassifier(strategy='stratified')
run_classification(dummy_stratified, X_train_scaled, X_test_scaled, y_train, y_test)


# 0.0.4 Decision Tree Classifier

# In[36]:


from sklearn.tree import DecisionTreeClassifier

descision_tree = DecisionTreeClassifier()
run_classification(descision_tree, X_train_scaled, X_test_scaled, y_train, y_test)


# 0.0.5 Gaussian

# In[37]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
run_classification(gaussian, X_train_scaled, X_test_scaled, y_train, y_test)


# 0.0.6 K Neighbors

# In[38]:


from sklearn.neighbors import KNeighborsClassifier
    
k_neighbors = KNeighborsClassifier()
run_classification(k_neighbors, X_train_scaled, X_test_scaled, y_train, y_test)


# ###Well, based on our results, all the models we tested had a better result
# ###than the baseline, what's good, but the difference is only 10% on avarage, so is not that impressive

# In[39]:


# Delete this variables to clean space
del X_test_scaled
del X_train_scaled


# ###Classification without the categorical columns

# In[40]:


# Divide in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature , target,
                                                    shuffle = True,
                                                    test_size=0.2,
                                                    random_state=1)


# In[42]:


from sklearn.preprocessing import OneHotEncoder
# Function to create the columns with the one hot encoding and concat with the original datasets

def one_hot_encoding(X_train, X_test, columns):
    one_hot_enc = OneHotEncoder()
    one_hot_enc.fit(X_train[columns])
    
    new_columns_train = one_hot_enc.transform(X_train[columns])
    new_columns_test = one_hot_enc.transform(X_test[columns])
    
    # Here we create loop all the columns we have and use them to create new names for each column,
    # being a name with the original name of the column concated with the label of the category
    all_column_names = np.array([])
    for index, column in enumerate(columns):
        column_names = column + one_hot_enc.categories_[index]
        all_column_names = np.concatenate((all_column_names, column_names), axis=None)

    temp = pd.DataFrame(new_columns_train.toarray(), columns=all_column_names, index = X_train.index)
    X_train = pd.concat([X_train, temp], axis=1)

    temp = pd.DataFrame(new_columns_test.toarray(), columns=all_column_names, index = X_test.index)
    X_test = pd.concat([X_test, temp], axis=1)
    
    X_train = X_train.drop(columns=object_column_names)
    X_test = X_test.drop(columns=object_column_names)
    
    return X_train, X_test


# In[43]:


# Encond the columns
X_train, X_test = one_hot_encoding(X_train, X_test, object_column_names)

# Scale the features
X_train_scaled, X_test_scaled = scaling(X_train, X_test)

# Delete the original variables
del X_train
del X_test


# ###The baseline don't change for the new processing, so we don't need to run it again!

# 0.0.8 Decision Tree Classifier

# In[44]:


descision_tree = DecisionTreeClassifier()
run_classification(descision_tree, X_train_scaled, X_test_scaled, y_train, y_test)


# In[45]:


gaussian = GaussianNB()
run_classification(gaussian, X_train_scaled, X_test_scaled, y_train, y_test)


# ###Classification without the categorical columns

# In[46]:


# Divide in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature , target,
                                                    shuffle = True,
                                                    test_size=0.2,
                                                    random_state=1)


# In[47]:


# Drop all the categorical columns without any previous processing
X_train = X_train.drop(columns = object_column_names)
X_test = X_test.drop(columns = object_column_names)

X_train_scaled, X_test_scaled = scaling(X_train, X_test)

del X_train
del X_test


# 0.0.9 Decision Tree Classifier

# In[48]:


descision_tree = DecisionTreeClassifier()
run_classification(descision_tree, X_train_scaled, X_test_scaled, y_train, y_test)


# 0.0.10 Gaussian

# In[49]:


gaussian = GaussianNB()
run_classification(gaussian, X_train_scaled, X_test_scaled, y_train, y_test)


# 0.0.11 K Neighbor

# In[50]:


k_neighbors = KNeighborsClassifier()
run_classification(k_neighbors, X_train_scaled, X_test_scaled, y_train, y_test)


# Well, this show us, that even without using the categorical columns we still have the same result
# as before if we don’t filter any of the columns we’re using on our ML models

# 0.0.12 Classification only using categorical columns

# In[51]:


# Divide in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature , target,
                                                    shuffle = True,
                                                    test_size=0.2,
                                                    random_state=1)


# In[52]:


drop_columns = ['DayOfWeek', 'Length', 'Flight', 'Time']
X_train = X_train.drop(columns=drop_columns)
X_test = X_test.drop(columns=drop_columns)


# In[53]:


X_train, X_test = one_hot_encoding(X_train, X_test, object_column_names)
X_train_scaled, X_test_scaled = scaling(X_train, X_test)

del X_train
del X_test


# 0.0.13 Decision Tree Classifier

# In[54]:


descision_tree = DecisionTreeClassifier()
run_classification(descision_tree, X_train_scaled, X_test_scaled, y_train, y_test)


# 0.0.14 Gaussian

# In[55]:


gaussian = GaussianNB()
run_classification(gaussian, X_train_scaled, X_test_scaled, y_train, y_test)


# Well, i think this proves that filtering our columns is super important to have good results in
# your classification. We had a better result just using the Airline encoding, so those columns were
# actually worsening our accuracy
