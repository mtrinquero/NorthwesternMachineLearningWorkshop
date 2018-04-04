from sklearn.datasets.samples_generator import make_blobs

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from numpy import random


#no need to go into details of this function
def draw_decision(X,y, model_list, model_description):
    # Plotting decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]),
                            model_list,model_description
                           ):

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                      s=20, edgecolor='k')
        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()



#####CLASSIFICATION EXAMPLE
# Generate sample data
centers = [[1, 1], [-1, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=0.9,
                            random_state=0)

#here is the situation
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('This is a classification task !')


#apply classifier on new data
X_new, not_used = make_blobs(n_samples=10, centers=[[1,1]], cluster_std=0.4,
                            random_state=0)

#here is the situation
plt.scatter(X[:,0], X[:,1], c=y)
plt.scatter(X_new[:,0], X_new[:,1], c='red')
plt.title('This is a classification task !')

#train classifier
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X, y)

#predict on new data
y_new = clf.predict(X_new)


###MANY DIFFERENT CLASSIFICATION TECHNIQUES


#Trees
#https://en.wikipedia.org/wiki/File:Titanic_Survival_Decison_Tree_SVG.png
#http://meru.cs.missouri.edu/courses/cecs401_data_mining/projects/group3/GermanCreditWeb.htm
clf1 = DecisionTreeClassifier(max_depth=1)
clf2 = DecisionTreeClassifier(max_depth=2)
model_list = [clf1, clf2]
clf1.fit(X, y)
clf2.fit(X, y)
model_description =  ['Decision Tree (depth=1)', 'Decision Tree (depth=2)']
draw_decision(X,y, model_list, model_description )


#kNN
#https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#/media/File:KnnClassification.svg
#OVERFITTING !
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = KNeighborsClassifier(n_neighbors=5)
clf3 = KNeighborsClassifier(n_neighbors=10)
clf4 = KNeighborsClassifier(n_neighbors=100)
model_list = [clf1, clf2, clf3, clf4]
clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)
model_description =  ['kNN, k=1', 'kNN, k=5', 'kNN, k=10', 'kNN, k=100']
draw_decision(X,y, model_list, model_description )


#logistic regression
#https://en.wikipedia.org/wiki/Logistic_regression
from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1.0)
model_list = [clf]
clf.fit(X, y)
model_description =  ['Logistic Regression']
draw_decision(X,y, model_list, model_description )

#SVM
#https://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
from sklearn import linear_model
clf1 = SVC(C=1,kernel='poly', degree=1, probability=True)
clf2 = SVC(C=0.001,kernel='rbf', probability=True)
clf3 = SVC(C=1,kernel='rbf', probability=True)
clf4 = SVC(C=1000, kernel='rbf', probability=True)
clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)
model_list = [clf1, clf2, clf3, clf4]
model_description =  ['Linear SVM', 'RBF SVM C=0.001', 'RBF SVM C=1', 'RBF SVM C=1000']
draw_decision(X,y, model_list, model_description )

#RF
#https://adpearance.com/blog/automated-inbound-link-analysis
from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators=1, max_depth = 2)
clf2 = RandomForestClassifier(n_estimators=5, max_depth = 2)
clf3 = RandomForestClassifier(n_estimators=10, max_depth = 2)
clf4 = RandomForestClassifier(n_estimators=1000, max_depth = 2)
clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)
model_list = [clf1, clf2, clf3, clf4]
model_description =  ['RF, 1 estimator' , 'RF, 5 estimators', 'RF, 10 estimators', 'RF, 1000 estimator']
draw_decision(X,y, model_list, model_description )


# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])
clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)
model_list = [clf1, clf2, clf3, eclf]
model_description =  ['Decision Tree (depth=4)', 'KNN (k=7)',
                             'Kernel SVM', 'Soft Voting']
draw_decision(X,y, model_list, model_description)

##TRAINING AND TESTING
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#training
clf_knn = KNeighborsClassifier(n_neighbors=1)
clf_knn.fit(X, y)

#I want to know the accuracy of my classifier
#Is this correct ?
y_predicted = clf_knn.predict(X)
print('Accuracy is: ' + str (100*accuracy_score(y, y_predicted)) + ' %')

#training/testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, marker = 'o')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker = 's', cmap='coolwarm')

#test set accuracy
clf_knn = KNeighborsClassifier(n_neighbors=10)
clf_knn.fit(X_train, y_train)
y_predicted = clf_knn.predict(X_test)
print('Accuracy is: ' + str (100*accuracy_score(y_test, y_predicted)) + ' %')

#test repeat
n_experiments = 100
list_of_accuracies = np.zeros(n_experiments)
for u in range(n_experiments):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn.fit(X_train, y_train)
    y_predicted = clf_knn.predict(X_test)
    print('Accuracy is: ' + str (100*accuracy_score(y_test, y_predicted)) + ' %')
    list_of_accuracies[u]=100*accuracy_score(y_test, y_predicted)
print('Mean accuracy: ' + str(np.mean(list_of_accuracies)) + ' standard deviation: ' + str(np.std(list_of_accuracies)))
print(' ')
print('6.385051e+139 possible choices of a test set')


##CROSS VALIDATION
#https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html
n_folds = 10
from sklearn.model_selection import KFold
kf = KFold(n_splits=n_folds, random_state=3)
y_predicted = np.zeros(y.shape)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn.fit(X_train, y_train)
    y_predicted[test_index] = clf_knn.predict(X_test)

print('Accuracy is: ' + str (100*accuracy_score(y, y_predicted)) + ' %')


#repeat cross-validation
random.seed(1)
from numpy import random
from sklearn.model_selection import KFold
n_reps=20
list_of_accuracies = np.zeros(n_reps)
for u in range(n_reps):
    n_folds = 10

    #randomly shuffle the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices] #dataset has been randomly shuffled


    kf = KFold(n_splits=n_folds)
    y_predicted = np.zeros(y_shuffled.shape)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
        y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]
        clf_knn = KNeighborsClassifier(n_neighbors=10)
        clf_knn.fit(X_train, y_train)
        y_predicted[test_index] = clf_knn.predict(X_test)

    print('Accuracy is: ' + str (100*accuracy_score(y_shuffled, y_predicted)) + ' %')
    list_of_accuracies[u] = 100*accuracy_score(y_shuffled, y_predicted)
print('Mean accuracy: ' + str(np.mean(list_of_accuracies)) + ' standard deviation: ' + str(np.std(list_of_accuracies)))

##REGRESSION

#https://statistics.laerd.com/spss-tutorials/linear-regression-using-spss-statistics.php
import numpy as np
from numpy import random

#plot example of linear relationship
X = np.random.uniform(-5,5,1000)
y = 2*X + 1 + np.random.randn(1000)
plt.figure()
plt.scatter(X,y, s=10)

#training a regression model
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(X.reshape(-1,1), y)
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
plt.figure()
plt.scatter(X,y, s=10)
plt.plot(X,regr.coef_*X + regr.intercept_ , color ='black')

#what about non-linearities
plt.close('all')
X = np.random.uniform(-5,5,1000).reshape(-1,1)
y = np.power(X,2) + 2*X + 1 + np.random.randn(1000).reshape(-1,1)
plt.figure()
plt.scatter(X,y, s=10)

#training a regression model with non-linearities
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
X_new = np.concatenate([X, np.power(X,2)] , axis=1)
regr.fit( X_new, y)
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
plt.figure()
plt.scatter(X,y, s=10)
plt.plot(np.arange(-5,5,0.01),regr.coef_[0][1]*np.power(np.arange(-5,5,0.01),2) +  regr.coef_[0][0]*np.arange(-5,5,0.01) + regr.intercept_ , color ='black')

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model

##REPEATED CROSS_VALIDATION

#plot example of linear relationship
X = np.random.uniform(-5,5,1000).reshape(-1,1)
y = 2*X.reshape(-1) + 1 + np.random.randn(1000)

random.seed(1)
from numpy import random
from sklearn.model_selection import KFold
n_reps=20
list_of_rmse = np.zeros(n_reps)
list_of_r2 =  np.zeros(n_reps)
for u in range(n_reps):
    n_folds = 10

    #randomly shuffle the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices] #dataset has been randomly shuffled


    kf = KFold(n_splits=n_folds)
    y_predicted = np.zeros(y_shuffled.shape)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
        y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_predicted[test_index] = regr.predict(X_test)

    print('RMSE is: ' + str (np.sqrt(mean_squared_error(y_shuffled, y_predicted))) )
    print('R squared is: ' + str(r2_score(y_shuffled, y_predicted)) )
    list_of_rmse[u] = np.sqrt(mean_squared_error(y_shuffled, y_predicted))
    list_of_r2[u] = r2_score(y_shuffled, y_predicted)

print('Mean rmse: ' + str(np.mean(list_of_rmse)) + ' standard deviation: ' + str(np.std(list_of_rmse) ) )
print('Mean R squared: ' + str(np.mean(list_of_r2)) + ' standard deviation: ' + str(np.std(list_of_r2)))


