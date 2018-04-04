import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#Loading dataset
wine_dataset = pd.read_csv('/media/polatalemdar/9E82D1BB82D197DB/Kaggle/wine dataset/winequality-red.csv')

#Look at dataset
print(wine_dataset.head())

#example indexing
print(wine_dataset.columns)

#features of the 3rd wine
print(wine_dataset.iloc[2])

#pH of 3rd wine sample
print(wine_dataset['pH'][2])

#add good vs bad wine as a target variable
wine_dataset['Good Wine'] = wine_dataset['quality']>5
print(wine_dataset.iloc[100])

#add some made up features
wine_dataset['Color Parameter'] = 5+np.random.randn(1599, 1)
wine_dataset['Fruitiness Parameter'] = 5+2*np.random.randn(1599, 1)

#seperate bad and good wine
bad_wine = wine_dataset[wine_dataset['Good Wine'] == False]
good_wine = wine_dataset[wine_dataset['Good Wine'] == True]

#plot the distribution of a feature
variable_to_plot = 'fixed acidity' # feature to plot
fig = plt.figure(figsize=(12,7)); # initialize your plot
ax1 = sns.kdeplot(bad_wine[variable_to_plot]  , shade=True, color="xkcd:cherry" , label="bad wine") #distribution of bad wines
ax2 = sns.kdeplot(good_wine[variable_to_plot]  , shade=True, color="xkcd:royal blue" , label="good wine") #distribution of good wines
#figure formatting
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.legend(loc=2, prop={'size': 16})
ax1.grid(color='grey', linestyle='--', linewidth=0.5)
plt.xlabel(variable_to_plot ,  fontsize=20)
plt.ylabel('Density', fontsize=20)


##...plot different features and see which ones are good...
##...select a set of features ...

#which features are good ??
for u in good_wine.columns:
    variable_to_plot = u # feature to plot
    fig = plt.figure(figsize=(6,4)); # initialize your plot
    ax1 = sns.kdeplot(bad_wine[variable_to_plot]  , shade=True, color="xkcd:cherry" , label="bad wine") #distribution of bad wines
    ax2 = sns.kdeplot(good_wine[variable_to_plot]  , shade=True, color="xkcd:royal blue" , label="good wine") #distribution of good wines
    #figure formatting
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.legend(loc=1, prop={'size': 12})
    ax1.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel(variable_to_plot ,  fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()


##TSNE Plots
#http://scikit-learn.org/stable/modules/manifold.html

#import stuff
from sklearn import manifold
from sklearn import preprocessing

#transform data using tsne
tsne = manifold.TSNE(n_components=2, init='pca' , method='barnes_hut' , random_state=4 , perplexity=60) #initialize tsne
list_features = ['fixed acidity', 'volatile acidity', 'citric acid',
       'chlorides',  'total sulfur dioxide', 'density',
        'sulphates', 'alcohol']
X = wine_dataset[list_features].values#these are your features (predictors)
y = wine_dataset['Good Wine'] #this is the target variable
X_standardized = preprocessing.scale(X) #standardize your data
X_tsne= tsne.fit_transform(X_standardized) #perform tsne

#tsne scatter plot
fig = plt.figure();
cm=plt.cm.get_cmap('jet') #set the colormap
pl=plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=y, s= 30, cmap= cm, alpha = 0.9 , edgecolors='none')
#figure formating
plt.title('TSNE Scatter Plot')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.xlabel('TSNE dimension 1', fontsize=12)
plt.ylabel('TSNE dimension 2', fontsize=12)
plt.colorbar(pl)
plt.tight_layout()

fig = plt.figure();
cm=plt.cm.get_cmap('jet') #set the colormap
pl=plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=wine_dataset['fixed acidity'].values, s= 30, cmap= cm, alpha = 0.9 , edgecolors='none')
#figure formating
plt.title(u)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.xlabel('TSNE dimension 1', fontsize=12)
plt.ylabel('TSNE dimension 2', fontsize=12)
plt.colorbar(pl)
plt.tight_layout()

##...look at different features with t-SNE...
##...does everything agree with the distribution plots ...


for u in list_features:
    fig = plt.figure();
    cm=plt.cm.get_cmap('jet') #set the colormap
    pl=plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=wine_dataset[u].values, s= 30, cmap= cm, alpha = 0.9 , edgecolors='none')
    #figure formating
    plt.title(u)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('TSNE dimension 1', fontsize=12)
    plt.ylabel('TSNE dimension 2', fontsize=12)
    plt.colorbar(pl)
    plt.tight_layout()


##What is standardization ?

#plot the distribution of a feature
variable_to_plot = 'alcohol' # feature to plot
var =wine_dataset[variable_to_plot].values
fig = plt.figure(figsize=(12,7)); # initialize your plot
ax1 = sns.kdeplot(var  , shade=True, color="xkcd:orange" ) #distribution of bad wines
ax2 = sns.kdeplot((var - np.mean(var))/np.std(var) , shade=True, color="xkcd:emerald") #distribution of good wines
#figure formatting
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.legend(loc=2, prop={'size': 16})
ax1.grid(color='grey', linestyle='--', linewidth=0.5)
plt.xlabel(variable_to_plot ,  fontsize=20)
plt.ylabel('Density', fontsize=20)

#Why standardize ?
print(wine_dataset.head())

#transform data using tsne
tsne = manifold.TSNE(n_components=2, init='pca' , method='barnes_hut' , random_state=4 , perplexity=60) #initialize tsne
list_features = ['fixed acidity', 'volatile acidity', 'citric acid',
       'chlorides',  'total sulfur dioxide', 'density',
        'sulphates', 'alcohol']
X = wine_dataset[list_features].values#these are your features (predictors)
y = wine_dataset['Good Wine'] #this is the target variable
X_tsne= tsne.fit_transform(X) #perform tsne

#tsne scatter plot
fig = plt.figure();
cm=plt.cm.get_cmap('jet') #set the colormap
pl=plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=y, s= 30, cmap= cm, alpha = 0.9 , edgecolors='none')
#figure formating
plt.title('TSNE Scatter Plot')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.xlabel('TSNE dimension 1', fontsize=12)
plt.ylabel('TSNE dimension 2', fontsize=12)
plt.colorbar(pl)
plt.tight_layout()



##Refine feature set
#transform data using tsne
tsne = manifold.TSNE(n_components=2, init='pca' , method='barnes_hut' , random_state=4 , perplexity=60) #initialize tsne
#list_features = ['chlorides',  'total sulfur dioxide', 'density']
list_features = ['fixed acidity', 'volatile acidity', 'citric acid',
        'sulphates', 'alcohol']
X = wine_dataset[list_features].values#these are your features (predictors)
y = wine_dataset['Good Wine'] #this is the target variable
X_standardized = preprocessing.scale(X) #standardize your data
X_tsne= tsne.fit_transform(X_standardized) #perform tsne

#tsne scatter plot
fig = plt.figure();
cm=plt.cm.get_cmap('jet') #set the colormap
pl=plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=y, s= 30, cmap= cm, alpha = 0.9 , edgecolors='none')
#figure formating
plt.title('TSNE Scatter Plot')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.xlabel('TSNE dimension 1', fontsize=12)
plt.ylabel('TSNE dimension 2', fontsize=12)
plt.colorbar(pl)
plt.tight_layout()


##...try to get a better t-SNE scatter plot ...
##... best possible t-SNE plot...
##...worst possible t-SNE plot...

##training and testing

##...discuss precision, recall, confusion matrix
#http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
#http://kflu.github.io/2016/08/26/2016-08-26-visualizing-precision-recall/

#import stuff
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

#for feature matrix and output vector
X = wine_dataset[['volatile acidity',
        'sulphates', 'alcohol']].values #whatever features you pick

y = wine_dataset[['Good Wine']].values.astype(int).reshape(-1)

#Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size  = 0.2 , random_state=1)

#standardize training and testing data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#define classifier
#clf = linear_model.LogisticRegression(C=1)
clf = RandomForestClassifier(n_estimators=100)
# clf = svm.SVC( C=100 , gamma=0.001,  kernel='rbf')
#clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.5, random_state=0)

#train on training set
clf.fit(X_train, y_train) #now you have a ML model !

#use classifier on testing data
y_predicted = clf.predict(X_test)

#asses model
conf_matrix = confusion_matrix(y_test, y_predicted)
accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)*100
precision = sklearn.metrics.precision_score(y_test, y_predicted)*100
recall = sklearn.metrics.recall_score(y_test, y_predicted)*100

#print scores
print("Accuracy is: " + str(accuracy) + "%")
print("Precision is: " + str(precision) + "%")
print("Recall is: " + str(recall) + "%")
print("Confision Matrix: ")
print(conf_matrix)



#Repeated Cross Validation
from sklearn.model_selection import KFold
from numpy import random
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

def perform_repeated_cv(X, y , model , n_reps, n_folds):
    #set random seed for repeartability
    random.seed(1)

    # perform repeated cross validation
    accuracy_scores = np.zeros(n_reps)
    precision_scores=  np.zeros(n_reps)
    recall_scores =  np.zeros(n_reps)

    for u in range(n_reps):

        print('Repetition Number: ' + str(u))

        #randomly shuffle the dataset
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices] #dataset has been randomly shuffled

        #initialize vector to keep predictions from all folds of the cross-validation
        y_predicted = np.zeros(y.shape)

        #perform 10-fold cross validation
        kf = KFold(n_splits=5 , random_state=142)
        for train, test in kf.split(X):

            #split the dataset into training and testing
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]
            y_test = y[test]

            #standardization
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            #train model
            clf = model
            clf.fit(X_train, y_train)

            #make predictions on the testing set
            y_predicted[test] = clf.predict(X_test)

        #record scores
        accuracy_scores[u] = accuracy_score(y, y_predicted)
        precision_scores[u] = precision_score(y, y_predicted)
        recall_scores[u]  = recall_score(y, y_predicted)

    #return all scores
    return accuracy_scores, precision_scores, recall_scores




#perfrom repeted CV with logistic regression
accuracy_scores, precision_scores, recall_scores = perform_repeated_cv(X, y
                                                 ,  RandomForestClassifier(n_estimators=100), n_reps = 10, n_folds= 10)
print('Results:')
print('Accuracy: ' + str(100*np.mean(accuracy_scores)) + '% +/- + ' + str(100*np.std(accuracy_scores)) )
print('Precision: ' + str(100*np.mean(precision_scores)) + '% +/- + ' + str(100*np.std(accuracy_scores)) )
print('Recall: ' + str(100*np.mean(recall_scores)) + '% +/- + ' + str(100*np.std(accuracy_scores)) )

##... try different classifiers and maximize accuracy:
#kNN
#Logistic Regression, l1, l2: try different C's
#Decision Tree, different max_depth, max_samples_leaf etc
#Random Forest, vary number of estimators
#Gradient Boosting, vary number of estimators, learning rate
#SVM with different kernels, vary the kernel and C
#Try different hyper-parameters


#plot results from the repetitions
fig, axes = plt.subplots(3, 1)
axes[0].plot(100*accuracy_scores , color = 'xkcd:cherry' , marker = 'o')
axes[0].set_xlabel('Repetition')
axes[0].set_ylabel('Accuracy Score (%)')

axes[1].plot(100*precision_scores , color = 'xkcd:royal blue' , marker = 'o')
axes[1].set_xlabel('Repetition')
axes[1].set_ylabel('Precision Score (%)')

axes[2].plot(100*precision_scores , color = 'xkcd:emerald' , marker = 'o')
axes[2].set_xlabel('Repetition')
axes[2].set_ylabel('Recall Score (%)')

plt.tight_layout()




#Regularization and hyper-parameter tuning
#https://en.wikipedia.org/wiki/Regularization_(mathematics)
from sklearn.linear_model import LogisticRegression

#set up the parameter sweep
c_sweep =  np.power(10, np.linspace(-4,4,100))

#perform repeated cross-validation by sweeping the parameter
accuracy_parameter_sweep = [] # keep scores here
std_parameter_sweep = [] #keep parameters in here
for c in c_sweep:

    #perform repeated cross-validation
    accuracy_scores, precision_scores, recall_scores = perform_repeated_cv(X, y ,
                                                                    LogisticRegression(C=c) , n_reps = 20, n_folds= 5 )

    ##append scores
    accuracy_parameter_sweep.append(np.mean(100*accuracy_scores))
    std_parameter_sweep.append(np.std(100*accuracy_scores))


#plot C vs. accuracy
plt.fill_between(c_sweep , np.array(accuracy_parameter_sweep) - np.array(std_parameter_sweep) ,
                 np.array(accuracy_parameter_sweep) + np.array(std_parameter_sweep) , facecolor = 'xkcd:light pink', alpha=0.7)
plt.semilogx(c_sweep,accuracy_parameter_sweep , color= 'xkcd:red' , linewidth=4)
plt.xlabel('C')
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression Accuracy vs. Hyper-parameter C')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()





#Number of estimators for a random forrest classifier
#set up the parameter sweep
n_est_sweep =  np.round(np.linspace(1,100,20)).astype(int)

#perform repeated cross-validation by sweeping the parameter
accuracy_parameter_sweep = [] # keep scores here
std_parameter_sweep = [] #keep parameters in here
for u in n_est_sweep:

    print(u)

    #perform repeated cross-validation
    accuracy_scores, precision_scores, recall_scores = perform_repeated_cv(X, y ,
                                                                    RandomForestClassifier(n_estimators=u), n_reps = 10, n_folds= 5)

    ##append scores
    accuracy_parameter_sweep.append(np.mean(100*accuracy_scores))
    std_parameter_sweep.append(np.std(100*accuracy_scores))


#plot C vs. accuracy
plt.fill_between(n_est_sweep , np.array(accuracy_parameter_sweep) - np.array(std_parameter_sweep) ,
                 np.array(accuracy_parameter_sweep) + np.array(std_parameter_sweep) , facecolor = 'xkcd:light pink', alpha=0.7)
plt.plot(n_est_sweep,accuracy_parameter_sweep , color= 'xkcd:red' , linewidth=4)
plt.xlabel('C')
plt.ylabel('Accuracy (%)')
plt.title('RF Accuracy vs. Hyper-parameter n_est')
plt.grid(True, which='both')
plt.tight_layout()






#Tuning k for kNN
from sklearn.neighbors import KNeighborsClassifier


#set up the parameter sweep
k_sweep =  np.round(np.linspace(1,40,20)).astype(int)

#perform repeated cross-validation by sweeping the parameter
accuracy_parameter_sweep = [] # keep scores here
std_parameter_sweep = [] #keep parameters in here
for u in k_sweep:

    print(u)

    #perform repeated cross-validation
    accuracy_scores, precision_scores, recall_scores = perform_repeated_cv(X, y ,
                                                                    KNeighborsClassifier(n_neighbors=u) , n_reps = 10, n_folds= 5)

    ##append scores
    accuracy_parameter_sweep.append(np.mean(100*accuracy_scores))
    std_parameter_sweep.append(np.std(100*accuracy_scores))


#plot k vs. accuracy
plt.fill_between(k_sweep , np.array(accuracy_parameter_sweep) - np.array(std_parameter_sweep) ,
                 np.array(accuracy_parameter_sweep) + np.array(std_parameter_sweep) , facecolor = 'xkcd:light pink', alpha=0.7)
plt.plot(k_sweep,accuracy_parameter_sweep , color= 'xkcd:red' , linewidth=4)
plt.xlabel('k')
plt.ylabel('Accuracy (%)')
plt.title('kNN Accuracy vs. k')
plt.grid(True, which='both')
plt.tight_layout()

