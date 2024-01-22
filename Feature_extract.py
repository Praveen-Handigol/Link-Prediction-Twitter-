# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:28:46 2023

@author: gess-icwar
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')
import os
import pandas as pd
#import pygrib
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import array as arr
os.chdir('C:/Users/gess-icwar/Desktop/Data Science')

print("Reading the raw data")
d = pd.read_csv('train.csv', header = None)

unq = d.stack().unique()

d1 = np.array(d)

id1 = d1[:,0]

n = 10000

#### For False ####

print("Working on the False Prediciton")
import random 
s1 = np.array(random.choices(id1, k=n))
s2 = np.array(random.choices(unq, k=n))

class1 = []
for iter1 in range(0, n):
    ind1 = np.array(np.where(id1 == s1[iter1]))

    row = d1[ind1,]

    def check_number_in_array(number, array):
        if number in array:
            return 1
        else:
            return 0

    is_present = check_number_in_array(s2[iter1], row)
    
    class1.append(is_present)


df1 = np.column_stack((s1,s2,class1))

df2 = df1[df1[:, 2] != 1]

df2 = df2[0:500,]

#### For True ####
print("Working on the True Prediciton")
n = 1000
s3 = np.array(random.choices(id1, k=n))
class2=[]
s4=[]

for iter2 in range(n):
    ind2 = np.array(np.where(id1 == s3[iter2]))
    row2 = d1[ind2,]
    row2 = row2.flatten()
    row2 = row2[~np.isnan(row2)]
    row2.shape[0]
   
    if row2.shape[0] > 1:
        s5 = np.array(random.choices(row2[1:], k=1))
        s4.append(s5)
        class2.append(1)
    else:
        out = np.array([0], dtype = 'float64')
        class2.append(0)
        s4.append(out)
    
    
s4=np.array(s4)

class2=np.array(class2)

df3 = np.column_stack((s3,s4,class2))

df4 = df3[df3[:, 2] != 0]

df4 = df4[0:500,]

ps = np.concatenate((df2, df4))

np.random.shuffle(ps)

#### Feature Extraction ####
## Source ID ##
following = []
follower = []
n1= 1000


for iter3 in range(0, n1):
    print("Working on the Feature Extraction on source id: "+ str(iter3))
    ps1 = ps[iter3,0]
    ## Following for source id ##
    ind3 = np.array(np.where(d1[:,0] == ps1))

    set1 = d1[ind3]

    set1 = set1[~np.isnan(set1)]

    following1 = set1.shape[0]-1
    
    following.append(following1)

    ## Follower for source id ##
    df5 = d1[:,1:]

    follower1 = np.count_nonzero(df5 == ps1)
    
    follower.append(follower1)
    
pars1 = np.column_stack((ps[0:n1,:],following,follower))  

## Reciever ID ##

following2 = []
follower2 = []
for iter3 in range(0, n1):
    print("Working on the Feature Extraction on reciever id: "+ str(iter3))
    ps1 = ps[iter3,1]
    ## Following for source id ##
    ind3 = np.array(np.where(d1[:,0] == ps1))

    set1 = d1[ind3]

    set1 = set1[~np.isnan(set1)]
    
    if set1.shape[0] > 1:
        following1 = set1.shape[0]-1
    else:
        following1 = int(0)
    
    following2.append(following1)

    ## Follower for source id ##
    df5 = d1[:,1:]

    follower1 = np.count_nonzero(df5 == ps1)
    
    follower2.append(follower1)
    
pars2 = np.column_stack((pars1,following2,follower2))  
    
    
## Transitivie Friends ##
tf2 = []

for iter3 in range(0, n1):
    print("Working on the transitive friends: "+ str(iter3))

    ps1 = ps[iter3,0]
    ps2 = ps[iter3,1]

    ind4 = np.array(np.where(d1[:,0] == ps1))
    set1 = d1[ind4]
    set1 = set1[~np.isnan(set1)]
    set1 = set1[1:]


    df5 = d1[:,1:]
    si1 = np.where(df5 == ps2)
    si2 = si1[0]
    siv1 = d1[si2,0]


    def find_intersection(arr1, arr2):
        set1 = set(arr1)
        set2 = set(arr2)
        intersection = set1.intersection(set2)
        return list(intersection)
    
    def find_union(arr1, arr2, arr3, arr4):
        set1 = set(arr1)
        set2 = set(arr2)
        set3 = set(arr3)
        set4 = set(arr4)
        union = set1.union(set2, set3, set4)
        return list(union)

    tf1 = len(find_intersection(set1,siv1))
    tf2.append(tf1)


pars3 = np.column_stack((pars2,tf2))  


## Common, Total Friends and Jaccard's Coefficient ##
cf2 = []
cf4 = []
tof2 = []
jc2 = []
cos2 = []
for iter3 in range(0, n1):
    print("Working on the common friends: "+ str(iter3))

    ps1 = ps[iter3,0]
    ps2 = ps[iter3,1]

    ind4 = np.array(np.where(d1[:,0] == ps1))
    set1 = d1[ind4]
    set1 = set1[~np.isnan(set1)]
    set1 = set1[1:]
    t1 = len(set1)

    ind5 = np.array(np.where(d1[:,0] == ps2))
    set2 = d1[ind5]
    set2 = set2[~np.isnan(set2)]
    set2 = set2[1:]
    t2 = len(set2)
    
    cf1 = len(find_intersection(set1,set2))
    cf2.append(cf1)
    
    df5 = d1[:,1:]
    si1 = np.where(df5 == ps1)
    si2 = si1[0]
    siv1 = d1[si2,0]
    t3 = len(siv1)
    
    si3 = np.where(df5 == ps2)
    si4 = si3[0]
    siv2 = d1[si4,0]
    t4 = len(siv2)
    
    cf3= len(find_intersection(siv1,siv2))
    cf4.append(cf3)
    
    tof1 = len(find_union(set1,set2,siv1,siv2))
    tof2.append(tof1)
    
       
    jc1 = (cf1+cf3)/(tof1)
    jc2.append(jc1)
    
    cos1 = (cf1+cf3)/((t1+t3)*(t2+t4))
    cos2.append(cos1)
    
pars4 = np.column_stack((pars3,cf2,cf4,tof2,jc2,cos2))  

ps_df = pd.DataFrame(pars4)

column_names = ['Source Node', 'Destination Node','Label', 'Out_d_source', 'in_d_source', 
                'out_d_dest', ' in_d_dest', 'Transitive_frnds','Out_d_common_frnds',
                'in_d_common_frnds', 'Total_frnds','Jaccard_coeff','cosine']

ps_df.columns = column_names

ps_df.to_csv('Trainset_features_10000.csv', index=False)

## Machine Learning Fitting ##

X = pars4[:,3:]
y = pars4[:,2]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(n_estimators = 10)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred = np.where(y_pred > 0.5, 1, 0)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Training Naive Bayes model on the Training set
from sklearn.naive_bayes import BernoulliNB
NaiveBayes_clf = BernoulliNB()
NaiveBayes_clf.fit(X_train, y_train)
y_pred_NB = NaiveBayes_clf.predict(X_test)
accuracy_score(y_test, y_pred_NB)

fprRF, tprRF, thresholdsRF = roc_curve(y_test, y_pred)
fprNB, tprNB, thresholdsNB = roc_curve(y_test, y_pred_NB)

roc_auc_RF = auc(fprRF, tprRF)
roc_auc_NB = auc(fprNB, tprNB)


fig, axes = plt.subplots(1,2)

# Plot on the first subplot
axes[0].plot(fprRF, tprRF, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_RF)
axes[0].set_title('Random Forest ROC')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc="lower right")


# Plot on the second subplot
axes[1].plot(fprNB, tprNB, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_NB)
axes[1].set_title('Naive Bayes ROC')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc="lower right")

# Adjust spacing between subplots
plt.tight_layout()

# Display the subplots
plt.show()

#### Calling test data ####

d_test = pd.read_csv('test.csv', header = None)
ps_test = np.array(d_test)

#### Feature Extraction ####
## Source ID ##
following_test = []
follower_test = []
n2 = 2000


for iter4 in range(0, n2):
    print("Working on the Feature Extraction on source id: "+ str(iter4))
    ps1_test = ps_test[iter4,0]
    ## Following for source id ##
    ind3_test = np.array(np.where(d1[:,0] == ps1_test))

    set1_test = d1[ind3_test]

    set1_test = set1_test[~np.isnan(set1_test)]

    following1_test = set1_test.shape[0]-1
    
    following_test.append(following1_test)

    ## Follower for source id ##
    df5_test = d1[:,1:]

    follower1_test = np.count_nonzero(df5 == ps1_test)
    
    follower_test.append(follower1_test)
    
pars1_test = np.column_stack((ps_test[0:n2,:],following_test,follower_test))  

## Reciever ID ##

following2_test = []
follower2_test = []
for iter4 in range(0, n2):
    print("Working on the Feature Extraction on reciever id: "+ str(iter4))
    ps1_test = ps_test[iter4,1]
    ## Following for source id ##
    ind3_test = np.array(np.where(d1[:,0] == ps1_test))

    set1_test = d1[ind3_test]

    set1_test = set1_test[~np.isnan(set1_test)]
    
    if set1_test.shape[0] > 1:
        following1_test = set1_test.shape[0]-1
    else:
        following1_test = int(0)
    
    following2_test.append(following1_test)

    ## Follower for source id ##
    df5 = d1[:,1:]

    follower1_test = np.count_nonzero(df5 == ps1_test)
    
    follower2_test.append(follower1_test)
    
pars2_test = np.column_stack((pars1_test,following2_test,follower2_test))  
    
    
## Transitivie Friends ##
tf2_test = []

for iter4 in range(0, n2):
    print("Working on the transitive friends: "+ str(iter4))

    ps1_test = ps_test[iter4,0]
    ps2_test = ps_test[iter4,1]

    ind4_test = np.array(np.where(d1[:,0] == ps1_test))
    set1_test = d1[ind4_test]
    set1_test = set1_test[~np.isnan(set1_test)]
    set1_test = set1_test[1:]


    df5 = d1[:,1:]
    si1_test = np.where(df5 == ps2_test)
    si2_test = si1_test[0]
    siv1_test = d1[si2_test,0]


    def find_intersection(arr1, arr2):
        set1 = set(arr1)
        set2 = set(arr2)
        intersection = set1.intersection(set2)
        return list(intersection)
    
    def find_union(arr1, arr2, arr3, arr4):
        set1 = set(arr1)
        set2 = set(arr2)
        set3 = set(arr3)
        set4 = set(arr4)
        union = set1.union(set2, set3, set4)
        return list(union)

    tf1_test = len(find_intersection(set1_test,siv1_test))
    tf2_test.append(tf1_test)


pars3_test = np.column_stack((pars2_test,tf2_test))  


## Common, Total Friends, Jaccard's Coefficient and Cosine Coefficient ##
cf2_test = []
cf4_test = []
tof2_test = []
jc2_test = []
cos2_test = []
for iter4 in range(0, n2):
    print("Working on the common friends: "+ str(iter4))

    ps1_test = ps_test[iter4,0]
    ps2_test = ps_test[iter4,1]

    ind4_test = np.array(np.where(d1[:,0] == ps1_test))
    set1_test = d1[ind4_test]
    set1_test = set1_test[~np.isnan(set1_test)]
    set1_test = set1_test[1:]
    t1_test = len(set1_test)

    ind5_test = np.array(np.where(d1[:,0] == ps2_test))
    set2_test = d1[ind5_test]
    set2_test = set2_test[~np.isnan(set2_test)]
    set2_test = set2_test[1:]
    t2_test = len(set2_test)
    
    cf1_test = len(find_intersection(set1_test,set2_test))
    cf2_test.append(cf1_test)
    
    df5 = d1[:,1:]
    si1_test = np.where(df5 == ps1_test)
    si2_test = si1_test[0]
    siv1_test = d1[si2_test,0]
    t3_test = len(siv1_test)
    
    si3_test = np.where(df5 == ps2_test)
    si4_test = si3_test[0]
    siv2_test = d1[si4_test,0]
    t4_test = len(siv2_test)
    
    cf3_test = len(find_intersection(siv1_test,siv2_test))
    cf4_test.append(cf3_test)
    
    tof1_test = len(find_union(set1_test,set2_test,siv1_test,siv2_test))
    tof2_test.append(tof1_test)
    
       
    jc1_test = (cf1_test+cf3_test)/(tof1_test)
    jc2_test.append(jc1_test)
    
    if ((t1_test+t3_test)*(t2_test+t4_test)) != 0:
        cos1_test = (cf1_test+cf3_test)/((t1_test+t3_test)*(t2_test+t4_test))
        cos2_test.append(cos1_test)
    else:
        cos1_test = 0
        cos2_test.append(cos1_test)
    
pars4_test = np.column_stack((pars3_test,cf2_test,cf4_test,tof2_test,jc2_test,cos2_test))  

ps_df_test = pd.DataFrame(pars4_test)

column_names = ['Source Node', 'Destination Node', 'Out_d_source', 'in_d_source', 
                'out_d_dest', ' in_d_dest', 'Transitive_frnds','Out_d_common_frnds',
                'in_d_common_frnds', 'Total_frnds','Jaccard_coeff','cosine']

ps_df_test.columns = column_names

ps_df_test.to_csv('Testset_features.csv', index=False)

### Test Prediction ####

X_testset = pars4_test[:,2:]
y_pred_NB_test = NaiveBayes_clf.predict(X_testset)

df_pred_NB_test = pd.DataFrame(y_pred_NB_test)

df_pred_NB_test.to_csv('Test_submission.csv', index=False)

# plt.figure()
# plt.plot(fprRF, tprRF, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_RF)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic ')
# plt.legend(loc="lower right")
# plt.show()
