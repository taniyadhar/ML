#!/usr/bin/env python
# coding: utf-8

# # Applied Machine Learning 

# # PROJECT DESCRIPTION

# Source of the dataset : Personal Loan Dataset has been taken from Kaggle.
# 
# Objective : Build a model that will help them identify the potential customers who have a higher probability of purchasing the loan.
# 
# Features : Age, Experience, Income, ZIPCode, Family, CCAvg, Education, Mortage, Personal Loan, Securities Account, CD Account, Online , CreditCard
# Target Value : PersonalLoan

# #### Column Description

# ID : Customer ID
# Age : Customer's age in completed years
# Experience : Number of years of professional experience
# Income : Annual income of the customer (USD 1000)
# ZIPCode : Home Address ZIP code.
# Family : Family size of the customer
# CCAvg : Avg. spending on credit cards per month (USD 1000)
# Education Education Level : 1: Undergrad; 2: Graduate; 3: Advanced/Professional
# Mortgage :  Value of house mortgage if any. (USD 1000)
# Personal Loan : Did this customer accept the personal loan offered in the last campaign?
# Securities Account : Does the customer have a securities account with the bank?
# CD Account : Does the customer have a certificate of deposit (CD) account with the bank?
# Online : Does the customer use internet banking facilities?
# CreditCard : Does the customer uses a credit card issued by UniversalBank?

# # CLASSIFICATION

# ## Import the required packages to build classification models
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#train and test split
from sklearn.model_selection import train_test_split

#Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#KNN Classification
from sklearn.neighbors import KNeighborsClassifier

#Logistic Regression
from sklearn.linear_model import LogisticRegression

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

#Confusion matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

#Support Vector Machine
from sklearn.svm import SVC, LinearSVC
from sklearn import svm

#Grid Search CV
from sklearn.model_selection import GridSearchCV

#Kernalize Support Vector
import matplotlib.gridspec as gridspec
import itertools

#Decision Boundary
from mlxtend.plotting import plot_decision_regions


#Project2 IMPORTANT lIBRARIES

#VotingClassifier
from sklearn.ensemble import VotingClassifier

#BaggingClassifier
from sklearn.ensemble import BaggingClassifier

get_ipython().run_line_magic('matplotlib', 'inline')

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

# PCA
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# ## Load the Personal Loan dataset in Python

# In[3]:


data = pd.read_csv('personalloan.csv')


# In[4]:


data.head(10) 


# In[5]:


data.shape


# # Detect Missing Values

# In[6]:


print((data[['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online',
       'CreditCard', 'PersonalLoan']] == '?').sum())


# In[7]:


data[['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online',
       'CreditCard', 'PersonalLoan']] = data[['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online',
       'CreditCard', 'PersonalLoan']].replace('?',np.NaN)


# In[8]:


data[['Experience']] = data[['Experience']].replace('?',np.NaN)


# In[9]:


print(data.isnull().sum())


# In[10]:


data = data.dropna()


# In[11]:


print(data.isnull().sum())


# In[12]:


data


# # Initial Data Analysis

# In[13]:


data.shape


# In[14]:


# Shape of training and test data set
def dataframe_shape(df):
    print("The dataframe has %d rows" %df.shape[0])
    print("The dataframe has %d columns" %df.shape[1])
dataframe_shape(data)


# In[15]:


# Columns/Feature in dataset
pd.DataFrame(data.columns,index=None,copy=False).T


# In[16]:


print(data.info())


# In[17]:


print(data.isnull().sum())


# In[18]:


sns.heatmap(data.isna(),yticklabels=False,cbar=False,cmap='magma')


# In[19]:


# There are no missing values in the dataset


# In[20]:


data.dtypes


# In[21]:


#Irregular value analysis


# In[22]:


data.describe().transpose()


# In[23]:


# change your column names
data.columns = [c.replace(' ', '_') for c in data.columns]


# In[24]:


data.columns


# In[25]:


# need to convert Securities Account, CD Account , Online, CreditCard,PersonalLoan from object to Binary 0 or 1


# In[26]:


data['Online'] = data['Online'].map({'Yes':1, 'No':0})


# In[27]:


data['CreditCard'] = data['CreditCard'].map({'Yes':1, 'No':0})


# In[28]:


data['PersonalLoan'] = data['PersonalLoan'].map({'Yes':1, 'No':0})


# In[29]:


data['CD_Account'] = data['CD_Account'].map({'Yes':1, 'No':0})


# In[30]:


data['Securities_Account'] = data['Securities_Account'].map({'Yes':1, 'No':0})


# In[31]:


data.dtypes


# In[32]:


# Columns/Feature in dataset
pd.DataFrame(data.columns,index=None,copy=False).T


# In[33]:


#finding unique data
data.apply(lambda x: len(x.unique()))


# The dataset contains 
# 4519:No ;   480 :Yes
# 
# 
# So, it is an Imbalanced Dataset

# # Exploratory Data Analysis

# In[34]:


sns.set_style('white')
sns.countplot(data=data,x='Education',hue='PersonalLoan',palette='RdBu_r')


# In[35]:


# Most of the loan applicants are professionals


# In[36]:


sns.barplot('Education','Mortgage',hue='PersonalLoan',data=data,palette='viridis',ci=None)
plt.legend(bbox_to_anchor=(1.2,1))


# In[37]:


# The value of house mortagage for the non-applicants is much lower than that of applicants. This could be a possible reason for them not applying for a loan or not finding a policy based on there need.


# In[38]:


sns.set_style('white')
sns.countplot(data=data,x='Securities_Account',hue='PersonalLoan',palette='Set2')


# It is clear that very few loan applicants have a securities account.

# In[39]:


sns.countplot(x='Family',data=data,hue='PersonalLoan',palette='Set1')


# Family size does not have any impact in personal loan. But,it seems families with size of 3 are more likely to take loan. 

# In[40]:


# datatypes present into training dataset

def datatypes_insight(data):
    display(data.dtypes.to_frame().T)
    data.dtypes.value_counts().plot(kind="barh")

datatypes_insight(data)


# ###### Bi-Variate Analysis - With Pivot Table for Catagorical variable:

# In[41]:


data[['CreditCard', 'PersonalLoan']].groupby(['CreditCard'], as_index=False).mean().sort_values(by='PersonalLoan', ascending=False)


# In[42]:


data[['Online', 'PersonalLoan']].groupby(['Online'], as_index=False).mean().sort_values(by='PersonalLoan', ascending=False)


# In[43]:


data[['Family', 'PersonalLoan']].groupby(['Family'], as_index=False).mean().sort_values(by='PersonalLoan', ascending=False)


# In[44]:


data[['Education', 'PersonalLoan']].groupby(['Education'], as_index=False).mean().sort_values(by='PersonalLoan', ascending=False)


# In[45]:


data[['CD_Account', 'PersonalLoan']].groupby(['CD_Account'], as_index=False).mean().sort_values(by='PersonalLoan', ascending=False)


# In[46]:


data[['Securities_Account', 'PersonalLoan']].groupby(['Securities_Account'], as_index=False).mean().sort_values(by='PersonalLoan', ascending=False)


# ## Feature correlation analysis

# In[47]:


feature = data.drop(["ID","PersonalLoan"],axis=1)
target = data["PersonalLoan"]


# In[48]:


corr = feature.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 10))
sns.heatmap(corr, mask=mask,annot=True,square=True)


# # Splitting Train and Test Set
# 

# In[49]:


# Feature name
X = pd.DataFrame(data, columns = ["Age","Experience","Income","ZIP_Code","Family","Education","Mortgage","Securities_Account","CD_Account","Online","CreditCard"])
X


# In[50]:


#Target name
y = pd.DataFrame(data, columns = ["PersonalLoan"])
y


# In[51]:


X_train_org , X_test_org, y_train, y_test = train_test_split(X, y, random_state = 0)


# # Scaling

# #### Scaling and Justification for type of scaling used
# 
# The scaling method we used is Standard Scaler as there is no outliers.
# Most of the values in each column is very small,the data would unnecessarily be skewed if we use min-max scaling.

# In[52]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# ### Classification Task:
# - Apply two voting classifiers - one with hard voting and one with soft voting
# - Apply any two models with bagging and any two models with pasting.
# - Apply any two models with AdaBoost boosting
# - Apply one model with gradient boosting
# - Apply PCA on data and then apply all the models in project 1 again on data you get from PCA. Compare your results with results in project 1. You don't need to apply all the models twice. Just copy the result table from project 1, prepare a similar table for all the models after PCA and compare both tables. Does PCA help in getting better results?
# - Apply deep learning models covered in class

# # Task 1: Two Voting Classifiers

# ### Hard Voting

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, precision_score, f1_score


# In[54]:


param_grid={'n_neighbors':range(1,10)}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, return_train_score=True, scoring='precision')
grid_search_knn.fit(X_train,y_train)

param_grid={'penalty':['l1','l2'],'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search_log = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True, scoring='precision')
grid_search_log.fit(X_train,y_train)

param_grid = {'C': [0.01, 10, 0.1, 1, 250, 100, 500]}
grid_search_SVC = GridSearchCV(SVC(kernel = 'rbf',probability = True),param_grid, cv=5, return_train_score=True,scoring='precision');
grid_search_SVC.fit(X_train, y_train)

voting_clf = VotingClassifier(
    estimators=[('lr', grid_search_log), ('knn', grid_search_knn), ('svc', grid_search_SVC)], voting='hard')
voting_clf.fit(X_train, y_train)


# In[55]:


from sklearn.metrics import precision_score
for clf in (grid_search_log, grid_search_knn, grid_search_SVC, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, precision_score(y_test, y_pred))


# ### Soft Voting

# In[56]:


from sklearn.model_selection import KFold
param_grid={'n_neighbors':range(1,10)}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, return_train_score=True, scoring='precision')
grid_search_knn.fit(X_train,y_train)
# k_test=cross_val_score(grid_search, X_test, y_test, cv=kfold,scoring='').mean()

param_grid={'penalty':['l1','l2'],'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search_log = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True, scoring='precision')
grid_search_log.fit(X_train,y_train)
# k_test=cross_val_score(grid_search, X_test, y_test, cv=kfold,scoring='').mean()

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 10, 0.1, 1, 250, 100, 500]}
grid_search_SVC = GridSearchCV(SVC(kernel= 'rbf',probability = True),param_grid, cv=5, return_train_score=True,scoring='precision');
grid_search_SVC.fit(X_train, y_train)

# test=cross_val_score(grid_search,X_test, y_test,cv=5,scoring='').mean();

voting_clf_soft = VotingClassifier(estimators=[('lr', grid_search_log), ('knn', grid_search_knn), ('svc', grid_search_SVC)],
                                   voting = 'soft')
voting_clf_soft.fit(X_train, y_train)


# In[57]:


for clf in (grid_search_log, grid_search_knn, grid_search_SVC, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, precision_score(y_test, y_pred))


# # Task 2 : Two models with Bagging and Pasting

# Bagging attempts to reduce the chance overfitting complex models
# 
# The main principle behind the ensemble model is that a group of weak learners come together to form a strong learner. Bagging (Bootstrap Aggregation) is used when our goal is to reduce the variance of a decision tree. Here idea is to create several subsets of data from training sample chosen randomly with replacement.

# ### Bagging 1: Logistic regression

# In[58]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[59]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100,200,250,500],'max_samples':[50,75,100,150,200]}

grid_search = GridSearchCV(BaggingClassifier(LogisticRegression(C=0.01, penalty='l1'),bootstrap=True,oob_score=True,random_state=0), param_grid, cv=3, return_train_score=True )
grid_search.fit(X_train,y_train)


# In[60]:


print("parameters",grid_search.best_params_)


# In[61]:


from sklearn.svm import LinearSVC
bag_class = BaggingClassifier(LinearSVC(C=250), n_estimators=100, max_samples=50, bootstrap=True,oob_score=True,random_state=0)
bag_class.fit(X_train,y_train)
btr_score = bag_class.score(X_train,y_train)
bte_score = bag_class.score(X_test,y_test)
oob = bag_class.oob_score_


# In[62]:


print(btr_score)
print(bte_score)
print(oob)


# ### Bagging 2:  SVC -RBF

# In[63]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100,200,250,500],'max_samples':[50,75,100,150,200]}

grid_search = GridSearchCV(BaggingClassifier(SVC(kernel='rbf',C=0.1,gamma= 0.1),bootstrap=True,oob_score=True,random_state=0), param_grid, cv=3, return_train_score=True )
grid_search.fit(X_train,y_train)


# In[64]:


print("parameters",grid_search.best_params_)


# In[65]:


from sklearn.svm import SVC
bag_class = BaggingClassifier(SVC(kernel='linear',C=250), n_estimators=100, max_samples=200, bootstrap=True,oob_score=True,random_state=0)
bag_class.fit(X_train,y_train)
btr_score = bag_class.score(X_train,y_train)
bte_score = bag_class.score(X_test,y_test)
oob = bag_class.oob_score_


# In[66]:


print(btr_score)
print(bte_score)
print(oob)


# # Pasting

# ### Pasting 1: SVC RBF Kernel

# In[67]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100,200,250,500],'max_samples':[50,75,100,150,200]}

grid_search = GridSearchCV(BaggingClassifier(SVC(kernel='rbf',C= 0.1, gamma = 0.1),bootstrap=False,random_state=0), param_grid, cv=3, return_train_score=True )
grid_search.fit(X_train,y_train)


# In[68]:


print("parameters",grid_search.best_params_)


# In[69]:


from sklearn.svm import SVC
bag_class = BaggingClassifier(SVC(kernel='rbf',C= 0.1, gamma = 0.1), n_estimators=100, max_samples=150, bootstrap=False,random_state=0)
bag_class.fit(X_train,y_train)
btr_score = bag_class.score(X_train,y_train)
bte_score = bag_class.score(X_test,y_test)


# In[70]:


print(btr_score)
print(bte_score)


# ### Pasting 2: Decision Tree

# In[71]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100,200,250,500],'max_samples':[50,75,100,150,200]}

grid_search = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(max_depth = 4),bootstrap=False,random_state=0), param_grid, cv=3, return_train_score=True )
grid_search.fit(X_train,y_train)


# In[72]:


print("parameters",grid_search.best_params_)


# In[73]:


from sklearn.tree import DecisionTreeClassifier
bag_class = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=150, bootstrap=False,random_state=0)
bag_class.fit(X_train,y_train)
btr_score = bag_class.score(X_train,y_train)
bte_score = bag_class.score(X_test,y_test)


# In[74]:


print(btr_score)
print(bte_score)


# # Task 3: 2 models with AdaBoost

# ### Adaboost 1: SVC Kernel Poly

# In[75]:


from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score , GridSearchCV

n_estimators_val = [10, 50, 100,200]
learning_rate_val = [0.1, 0.5, 1]

param_grid = dict(n_estimators = n_estimators_val, learning_rate=learning_rate_val)

ada_reg = AdaBoostClassifier(SVC(kernel='poly',C= 500, degree = 1),algorithm="SAMME", random_state=0)

adaboost_classifier = GridSearchCV(estimator=ada_reg, param_grid = param_grid )
adaboost_classifier.fit(X_train, y_train)
print ('Best Parameters: ',adaboost_classifier.best_params_)


# In[76]:


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(SVC(kernel='poly',C= 500, degree = 1), n_estimators=100, algorithm="SAMME", learning_rate=1, random_state=0)
ada_clf.fit(X_train, y_train)


# In[77]:


adtr_score = ada_clf.score(X_train,y_train)
adte_score = ada_clf.score(X_test,y_test)


# In[78]:


print(adtr_score)
print(adte_score)


# ### Adaboost 2: Decision Tree Classifier

# In[79]:


from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score , GridSearchCV

n_estimators_val = [10, 50, 100,200]
learning_rate_val = [0.1, 0.5, 1]

param_grid = dict(n_estimators = n_estimators_val, learning_rate=learning_rate_val)

ada_classifier = AdaBoostClassifier(DecisionTreeClassifier(),algorithm="SAMME.R", random_state=0)

adaboost_classifier = GridSearchCV(estimator=ada_reg, param_grid = param_grid )
adaboost_classifier.fit(X_train, y_train)
print ('Best Parameters: ',adaboost_classifier.best_params_)


# In[80]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100, algorithm="SAMME.R",learning_rate=1, random_state=0)
ada_clf.fit(X_train, y_train)
adtr_score = ada_clf.score(X_train,y_train)
adte_score = ada_clf.score(X_test,y_test)


# In[81]:


print(adtr_score)
print(adte_score)


# # Task 4: Gradient Boosting

# In[82]:


from sklearn.ensemble import GradientBoostingClassifier

n_estimators_val = [50, 100, 200, 500]
learning_rate_val = [0.01, 0.1, 0.5, 1]
max_depth_val = [1, 2, 3, 4]

param_grid = dict(n_estimators = n_estimators_val, learning_rate=learning_rate_val, max_depth=max_depth_val)
gbrg = GradientBoostingClassifier(random_state=0)

gradientboost_classifier = GridSearchCV(estimator=gbrg, param_grid=param_grid)
gradientboost_classifier.fit(X_train, y_train)

print (gradientboost_classifier.best_params_)
print('Train score: {:.2f}'.format(gradientboost_classifier.score(X_train, y_train)))
print('Test score: {:.2f}'.format(gradientboost_classifier.score(X_test, y_test)))


# # Additional Model- Random Forest 

# In[83]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [25,50,100,200,300,400,500]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, return_train_score=True,scoring='precision');

grid_search.fit(X_train, y_train)

test=cross_val_score(grid_search,X_test, y_test,cv=5,scoring='precision').mean();


# In[84]:


print("Train:",grid_search.best_score_)
print("Test",test)
print("parameters",grid_search.best_params_)


# # Classification Post PCA

# In[85]:


from sklearn.decomposition import PCA

pca = PCA(n_components=0.97)
pca.fit(X_train)
pca.n_components_


# In[86]:


X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# In[87]:


from sklearn.neighbors import KNeighborsClassifier
kfold =KFold(n_splits=5, random_state=0)

param_grid={'n_neighbors':range(1,10)}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kfold, return_train_score=True, scoring='precision')
grid_search.fit(X_train,y_train)

k_test=cross_val_score(grid_search, X_test, y_test, cv=kfold,scoring='precision').mean()


# In[88]:


print("Train:",grid_search.best_score_)
print("Test",k_test)
print("parameters",grid_search.best_params_)


# ### Adding results to post_pca_class(post_pca_classification_results)

# In[89]:


post_pca_class = pd.DataFrame(columns=('S.No','Model_Name','Parameters', 'Train_Score', 'Test_Score'))


# In[90]:


post_pca_class.loc[len(post_pca_class)]=[1,'KNN Classifier',grid_search.best_params_,grid_search.best_score_,k_test]


# In[91]:


post_pca_class


# In[92]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix
predicted = cross_val_predict(grid_search,X_test,y_test, cv=kfold)
confusion = confusion_matrix(y_test, predicted)


# In[93]:


from sklearn.metrics import classification_report
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, predicted, target_names=["No", "Yes"]))


# # Logistic Model

# In[94]:


from sklearn.linear_model import LogisticRegression
kfold =KFold(n_splits=5, random_state=0)
param_grid={'penalty':['l1','l2'],'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, return_train_score=True, scoring='precision')
grid_search.fit(X_train,y_train)

k_test=cross_val_score(grid_search, X_test, y_test, cv=kfold,scoring='precision').mean()


# In[95]:


print("Train:",grid_search.best_score_)
print("Test",k_test)
print("parameters",grid_search.best_params_)


# In[96]:


post_pca_class.loc[len(post_pca_class)]=[2,'Logistic Model',grid_search.best_params_,grid_search.best_score_,k_test]


# In[97]:


post_pca_class


# In[98]:


from sklearn.metrics import confusion_matrix
predicted = cross_val_predict(grid_search,X_test,y_test, cv=kfold)
confusion = confusion_matrix(y_test, predicted)


# In[99]:


from sklearn.metrics import classification_report
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, predicted, target_names=["No", "Yes"]))


# # Linear SVC

# In[100]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 10, 0.1, 1, 250, 100, 500]}
grid_search = GridSearchCV(LinearSVC(),param_grid, cv=kfold, return_train_score=True,scoring='precision');

grid_search.fit(X_train, y_train)

test=cross_val_score(grid_search,X_test, y_test,cv=5,scoring='precision').mean();


# In[101]:


print("Train:",grid_search.best_score_)
print("Test",test)
print("parameters",grid_search.best_params_)


# In[102]:


post_pca_class.loc[len(post_pca_class)]=[3,'Linear SVC',grid_search.best_params_,grid_search.best_score_,test]


# In[103]:


post_pca_class


# In[104]:


from sklearn.metrics import confusion_matrix
predicted = cross_val_predict(grid_search,X_test,y_test, cv=kfold)
confusion = confusion_matrix(y_test, predicted)


# In[105]:


from sklearn.metrics import classification_report
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, predicted, target_names=["No", "Yes"]))


# # SVC - Kernel Trick

# In[106]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 10, 0.1, 1, 250, 100, 500]}
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=kfold, return_train_score=True,scoring='precision');

grid_search.fit(X_train, y_train)

test=cross_val_score(grid_search,X_test, y_test,cv=5,scoring='precision').mean();


# In[107]:


print("Train:",grid_search.best_score_)
print("Test",test)
print("parameters",grid_search.best_params_)


# In[108]:


post_pca_class.loc[len(post_pca_class)]=[4,'SVC - Kernel Trick',grid_search.best_params_,grid_search.best_score_,test]


# In[109]:


post_pca_class


# In[110]:


from sklearn.metrics import confusion_matrix
predicted = cross_val_predict(grid_search,X_test,y_test, cv=kfold)
confusion = confusion_matrix(y_test, predicted)


# In[111]:


from sklearn.metrics import classification_report
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, predicted, target_names=["No", "Yes"]))


# In[112]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 10, 0.1, 1, 250, 100, 500],'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, return_train_score=True,scoring='precision');

grid_search.fit(X_train, y_train)

test=cross_val_score(grid_search,X_test, y_test,cv=5,scoring='precision').mean();


# In[113]:


print("Train:",grid_search.best_score_)
print("Test",test)
print("parameters",grid_search.best_params_)


# In[114]:


post_pca_class.loc[len(post_pca_class)]=[5,'SVC - RBF',grid_search.best_params_,grid_search.best_score_,test]


# In[115]:


post_pca_class


# In[116]:


from sklearn.metrics import confusion_matrix
predicted = cross_val_predict(grid_search,X_test,y_test, cv=kfold)
confusion = confusion_matrix(y_test, predicted)


# In[117]:


from sklearn.metrics import classification_report
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, predicted, target_names=["No", "Yes"]))


# In[118]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 10, 0.1, 1, 250, 100, 500],'degree':[1,2,3,4]}
grid_search = GridSearchCV(SVC(kernel='poly'), param_grid, cv=5, return_train_score=True,scoring='precision');

grid_search.fit(X_train, y_train)

test=cross_val_score(grid_search,X_test, y_test,cv=5,scoring='precision').mean();


# In[119]:


print("Train:",grid_search.best_score_)
print("Test",test)
print("parameters",grid_search.best_params_)


# In[120]:


post_pca_class.loc[len(post_pca_class)]=[6,'Linear SVC - Polynomial Kernel',grid_search.best_params_,grid_search.best_score_,test]


# In[121]:


post_pca_class


# In[122]:


from sklearn.metrics import confusion_matrix
predicted = cross_val_predict(grid_search,X_test,y_test, cv=kfold)
confusion = confusion_matrix(y_test, predicted)


# In[123]:


from sklearn.metrics import classification_report
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, predicted, target_names=["No", "Yes"]))


# # Decision Tree

# In[124]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X_train, y_train)
dt_tr_sc=cross_val_score(tree, X_train, y_train, cv=5,scoring ='precision').mean()
dt_te_sc=cross_val_score(tree, X_test, y_test, cv=5,scoring='precision').mean()
print(dt_tr_sc)
print(dt_te_sc)


# In[125]:


post_pca_class.loc[len(post_pca_class)]=[7,'Decision Tree Classifier',"None",dt_tr_sc,dt_te_sc]


# In[126]:


post_pca_class


# In[127]:


from sklearn.metrics import confusion_matrix
predicted = cross_val_predict(grid_search,X_test,y_test, cv=kfold)
confusion = confusion_matrix(y_test, predicted)


# In[128]:


from sklearn.metrics import classification_report
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, predicted, target_names=["No", "Yes"]))


# In[129]:


post_pca_class


# # Does PCA help in getting better results?
# 
# #1.Before PCA in Project 1, the best model was the Decision Tree Classifier Decision Tree Classifier Train score of 0.98 and Test score of 1.00
# #2.After PCA, the best model was KNN with a train score of 0.96 and a test score of 0.93.
# #3.There seems to be a marginal improvement in some of the models after PCA but there also seems to be a decrease in some models. So we cannot conlusively say that PCA helps in getting better results.

# # 1. Perceptron

# In[130]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# In[131]:


#step 1: build model
model1 = Sequential()
#input layer
model1.add(Dense(10, input_dim = 10, activation = 'relu'))
#hidden layers
#output layer
model1.add(Dense(1, activation = 'sigmoid'))
#step 2: make computational graph - compile
model1.compile(loss= 'binary_crossentropy' , optimizer = 'adam',metrics = ['accuracy'] )


# In[132]:


#step 3: train the model - fit
model1.fit(X_train, y_train, epochs = 75, batch_size = 200);


# ### Perceptron Evaluation Metrics

# In[133]:


model1.evaluate(X_train, y_train)


# In[134]:


model1.evaluate(X_test, y_test)


# ### Perceptron Prediction Score

# In[135]:


train_pred=model1.predict_classes(X_train)
pred=model1.predict_classes(X_test)


# In[136]:



print("train",precision_score(y_train, train_pred))
print("test",precision_score(y_test,pred))


# # 2.Multi-Level Perceptron

# In[137]:


#step 1: build model
model1 = Sequential()
#input layer
model1.add(Dense(30, input_dim = 10, activation = 'relu'))
#hidden layers
model1.add(Dense(20, activation = 'relu'))
model1.add(Dense(10, activation = 'relu'))
#output layer
model1.add(Dense(1, activation = 'sigmoid'))

#step 2: make computational graph - compile
model1.compile(loss= 'binary_crossentropy' , optimizer = 'adam',metrics = ['accuracy'] )

#step 3: train the model - fit
model1.fit(X_train, y_train, epochs = 100, batch_size = 200);


# ### Multi Layer Perceptron Evaluation Metrics

# In[138]:


model1.evaluate(X_train, y_train)


# In[139]:


model1.evaluate(X_test, y_test)


# ### Multi Layer Perceptron Precision Scores

# In[140]:


train_pred=model1.predict_classes(X_train)
pred=model1.predict_classes(X_test)


# In[141]:


print("train",precision_score(y_train, train_pred))
print("test",precision_score(y_test,pred))


# # Conclusion

# #Task1.Two Voting Classifiers(with models KNN,Logistic,RBF Kernel)- One with hard voting and one with soft voting were applied.
# Hard Voting Classifier Output was 1.00 and Soft Voting Classifier was 0.963
# #Task2.Two models with bagging(Logistic Regression,SVC RBF) and two models with pasting(SVC RBF Kernel,Decision Tree Classifier) were applied.
# #Task3.Two models with adaboost boosting(SVC Polynomial Kernel Trick,Decision Tree Classifier) were applied.
# #Task4.One model with gradient boosting was applied.
# #Task5.The following models were applied before PCA and after PCA:
#   #a.KNN Classifier
#   #b.Logistic Regression
#   #c.Linear SVC
#   #d.SVC - 'Linear' Kernel Trick
#   #e.SVC - 'Polynomial' Kernel Trick
#   #f.SVC - 'RBF' Kernel Trick
#   #g.Decision Tree Classifier
#   #The results were compared and the explanation of whether PCA yields better results was given.
# #Task6.Two Deep Learning models( Perceptron and Multi Layer Perceptron) were applied.
