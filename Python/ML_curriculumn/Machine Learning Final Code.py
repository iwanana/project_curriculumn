########################### Loading library ##################################

##loading library: 
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

################### Loading date and basic data transformation ####################
data=pd.read_csv('adult.csv')
#replace question marks with python-recognizable NaN
data = data.replace(' ?', np.NaN)

##check missing value:
missing_value_column=data.columns[data.isna().any()].tolist()
data[missing_value_column].isnull().sum(axis=0)  ##checking missing value variable


##drop all the rows contain missing value ( missing value ratio around 5%)
data=data.dropna()
label = data['target']

##Data Exploration: 
#import seaborn as sns
#import matplotlib.pyplot as plt
#data['target'].value_counts().plot(kind='bar')
##from the table we learn that the data set is very unbalaced,we will need use F1 score to #measure the performance of the model

##Data visualizaton for categorical varaibles: 
#data2=data.select_dtypes(include=['object'])
#data2.head()
#for i in range(data2.shape[1]-1):
   #a=data2.groupby([data2.columns[i],'target'])[data2.columns[i]].count().unstack('target').fillna(0)
    #a.plot(kind='bar', stacked=True)
    #print(a)

#data visualization for numerical variables: 
#data3=data.select_dtypes(include=['int64'])
#data3['target']=data2['target']
#data3.head()
#import seaborn as sns
#df = sns.load_dataset('iris')

##age vs educatio-number:
#sns.lmplot( x="age", y="education-num", data=data3, fit_reg=False, hue='target', #legend=False)
#plt.legend(loc='lower right')

##age vs hours-per-week: 
#sns.lmplot( x="age", y="hours-per-week", data=data3, fit_reg=False, hue='target', #legend=False)
#plt.legend(loc='lower right')

##hours vs education-number: 
#sns.lmplot( x="education-num", y="hours-per-week", data=data3, fit_reg=False, hue='target', #legend=False)
#plt.legend(loc='lower right')


#print(len(data3))
#print((data3['capital-gain'] == 0).sum())
#print((data3['capital-loss'] == 0).sum())

##based on the data visualization, we should drop 'native-country' variable since it has no #predictive power: 
data=data.drop('native-country',axis=1)

# data2.columns
# 'marital-status'
# 'relationship'
# 'race'

#tester1=data2.groupby(['marital-status','target']).size()
#print(tester1.groupby(level=0).apply(lambda x: x / x.sum()))

#tester2=data2.groupby(['relationship','target']).size()
#print(tester2.groupby(level=0).apply(lambda x: x / x.sum()))

#tester3=data2.groupby(['race','target']).size()
#print(tester3.groupby(level=0).apply(lambda x: x / x.sum()))

#tester4=data2.groupby(['race','target']).size()
#print(tester4.groupby(level=0).apply(lambda x: x / x.sum()))

##Data spliting: Train&Test set 
data=data.drop('target',axis=1)

X_train, X_test, y_train, y_test = train_test_split( data, label, test_size=0.33, random_state=42)
y_train=pd.get_dummies(y_train)   
y_test=pd.get_dummies(y_test)   

X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)

##make sure test and train set have the same number of columns: 
missing_column=set(X_train.columns).symmetric_difference(set(X_test.columns))
missing_column=list(missing_column)
adding=pd.DataFrame(0, index=np.arange(len(X_test)), columns=missing_column)
X_test=X_test.reset_index(drop=True)
X_test=X_test.join(adding, how='outer')
X_test=X_test[X_train.columns]


y_train=y_train.as_matrix()[:,1] 
y_test=y_test.as_matrix()[:,1] 


column_name=X_train.columns
X_train=X_train.as_matrix()
X_test=X_test.as_matrix()


###########################Final Test file######################
##This is a seperate test file: used for final model validation 
final_test=pd.read_csv('adult_test.csv')
final_test_y=final_test['target']
final_test_x=final_test.drop(['target','native-country'],axis=1)
final_test_y=pd.get_dummies(final_test_y)
final_test_x=pd.get_dummies(final_test_x)
final_test_y=final_test_y.as_matrix()[:,1] 
final_test_x=final_test_x[column_name]

final_test_x=final_test_x.as_matrix()
final_test_x=final_test_x.astype('float32')

######################## PCA with numpy and scipy ####################

def PCA(data, dims_rescaled_data=2):
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs =np.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T


X_train=X_train.astype('float32')
X_train=PCA(X_train, 100)

X_test=X_test.astype('float32')
X_test=PCA(X_test, 100)

final_test_x=final_test_x.astype('float32')
final_test_x=PCA(final_test_x, 100)

##############################################################################################################
#######################################SVM MODEL##############################################################
def learnWbL(X,y,Xtest,ytest,epsi,lam): # in this input y is 0/1
    
  ### DATA PRELIMINARY ###
  # append initial b multiplier
    X = np.c_[X,[1]*len(X)]
    Xtest = np.c_[Xtest,[1]*len(Xtest)]
  # fix y and use 1/-1 instead
  # train y
    class_train = []
    for i in y:
        if i==1:
            class_train.append(1)
        else:
            class_train.append(-1)   
    y = np.array(class_train) 
  # test y
    class_test = []
    for i in ytest:
        if i==1:
            class_test.append(1)
        else:
            class_test.append(-1)   
    ytest = np.array(class_test)  
    
  # initial values
    w = np.random.normal(0,0.1,len(X[1,:])) # randomly set initial w
    #print(np.round(w,4))
    argmin = 10e+100 # a large enough argmin so that the loop won't end at the first round
    k=0 # count iteration number (from begining to end of the train set)
    gma = 0.5 # set our gamma and modify after we have w updated
    accuracy1 = 0 # small initial accuracy so that the loop won't end at the first round
    
  ### START LEARNING ###
  # start point timestamp
    t1 = time.time()
    #print(t1)
    
    while True:
  # update w
        for i in range(len(X)):
            w -= epsi *( w-gma * (y[i]*X[i]))+2*w/lam
            #for n in range(len(w)):
                
                #if (y[i]*np.dot(X[i],w)<1):
                #w -= epsi *( w-gma0 * (y[i]*X[i]))
                #w[n] -= epsi *( w[n]-gma0 * (y[i]*X[i][n])+ abs(w[n])/vlam[n]) 

                #print(w,np.dot(w.T,w) + gma*(np.dot((1-y),(np.dot(X,w)+1)) + np.dot(y,(1-np.dot(X,w)))))

  # update gamma
        for m in range(len(X)):
            if ((y[m]==1 and abs(np.dot(X[m],w))<1) or (y[m]==-1 and abs(np.dot(X[m],w))<1)):
                gma += epsi*( (1+y[m])/2*y[m]*(1-np.dot(X[m],w)) + (1-y[m])/2*y[m]*(np.dot(X[m],w)+1) )
# ============================================================================= 
#               
#  FORMULA EXPLAINATION:
#       Since y[m] is either 1 or -1, if the data point has y=1, the first item will remain 
#       and the second one will disappear due to 1-y=0. In the contrast, if y=-1, the second
#       term will be added.
#               
# =============================================================================
        
  # get new objective function value for further compare
        arg = np.dot(w.T,w) + gma*(np.dot((1-y)/2,(np.dot(X,w)+1)) + np.dot((1+y)/2,(1-np.dot(X,w)))) + np.dot(w.T,w)/lam  # initial argmin value


  # calculate accuracy
        c=0 # correctly labelled counter  
        for j in range(len(Xtest)):       
            if (np.dot(Xtest[j],w)>1 and ytest[j]==1) or (np.dot(Xtest[j],w)<-1 and ytest[j]==-1):
                c +=1
        accuracy2 = c/len(Xtest)
        
  # quit criteria check 
# =============================================================================
# 
#          NOTE: 
#              We consider both accuracy and objective function value.
#              Several loops will not stop after 200 or more iteration, 
#              with it's accuracy not change any more. We force to break the 
#              loop if the iteration is 200.
# 
# =============================================================================
        if (accuracy2<=accuracy1 and arg >= argmin) or (k==200):
            ### END LEARNING ###
            # end point timestamp 
            t2 = time.time()
            #print(t2)
            break
        else:
            argmin = arg
            accuracy1 = accuracy2
            k +=1
            
            
    return k,round(accuracy1,6),round(t2-t1,2),w

# choose hyper parameters from the list below
epsilist = np.array([0.000001,0.00001,0.0001,0.001,0.01])
lamlist = np.array([10,100,1000,10000])

### LIST and COMPARE the running result
# epsilon, lamhda value
# combination number (running round)
# output the combination that gives the highest accuracy of the training-test set
def smartchoose(epsilist,lamlist):
    i=0
    rate0 = 0
    for epsi in epsilist:
        for lam in lamlist:
            i +=1
            print('epsilon=',epsi,',lamhda=',lam,',combination track=',i)
            try:
                k,rate,t,wout = learnWbL(X_train,y_train,X_test,y_test,epsi,lam) 
                print('accuracy=',rate,',running time=',t)
                if rate>rate0:
                    tracki = i
                    rate0 = rate                
            except:
                print('This combination is not workable.')
                pass
    return tracki
            
smartchoose(epsilist,lamlist)
# the result turns out that when epsilon=0.01 and lamhda=100, accuracy is the highest
# now we calculate the best w
k,rate,t,wout = learnWbL(X_train,y_train,X_test,y_test,0.01,100)



#############################################################################################################
n,m = final_test_x.shape # for generality
X0 = np.ones((n,1))
final_test_x = np.hstack((final_test_x,X0))


################################################
preditve_result=np.dot(final_test_x,wout)
b = preditve_result >=1
predict_result=b.astype(int)
df_confusion = pd.crosstab(final_test_y, predict_result)
df_confusion
#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(final_test_y, predict_result,labels=[1, 0])
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)


# make a plot by plt
import seaborn as sns; sns.set()
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(final_test_y,predict_result))
print(confusion_matrix(final_test_y,predict_result))

mat = confusion_matrix(final_test_y,predict_result)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')



################################# Package PCA + SVM/Neural Networks ###############################################

################### Loading date and basic data transformation ####################
data=pd.read_csv('adult.csv')
#replace question marks with python-recognizable NaN
data = data.replace(' ?', np.NaN)

##check missing value:
missing_value_column=data.columns[data.isna().any()].tolist()
data[missing_value_column].isnull().sum(axis=0)  ##checking missing value variable


##drop all the rows contain missing value ( missing value ratio around 5%)
data=data.dropna()
label = data['target']

data=data.drop('native-country',axis=1)

data=data.drop('target',axis=1)

X_train=pd.get_dummies(data)
y_train=pd.get_dummies(label)

column_name=X_train.columns

############ Train Set ##############################################
X_train=X_train.as_matrix()
y_train=y_train.as_matrix()[:,-1] 

########################### Test Set ######################
##This is a seperate test file: used for final model validation 
final_test=pd.read_csv('adult_test.csv')
final_test_y=final_test['target']
final_test_x=final_test.drop(['target','native-country'],axis=1)
final_test_y=pd.get_dummies(final_test_y)
final_test_x=pd.get_dummies(final_test_x)
y_test=final_test_y.as_matrix()[:,1] 
X_test=final_test_x[column_name]


######################################################## PCA #################################################
from sklearn.decomposition import PCA
X_train=X_train.astype('float32')
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
X_train = pca.transform(X_train)

X_test=X_test.astype('float32')
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_test)
X_test = pca.transform(X_test)


############################################ Neural Networks ################################################
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='sgd', activation = 'logistic', alpha=1e-5, random_state=1)
clf = clf.fit(X = X_train, y = y_train) 
pred_train_NeuralNetwork = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_train_NeuralNetwork)
## Accuracy: 0.7723112830907193 ###############

### Visualization ###

import seaborn as sns; sns.set()
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred_train_NeuralNetwork))
print(confusion_matrix(y_test,pred_train_NeuralNetwork))

mat = confusion_matrix(y_test, pred_train_NeuralNetwork)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')

################################################# SVM ########################################################
############ 6 mins running time! #################
from sklearn import svm
clf = svm.SVC()
clf.fit(X = X_train, y = y_train) 
pred_train_svm = clf.predict(X_test)
accuracy_score(y_test, pred_train_svm)
## 
#0.7637123026841103 ##############












