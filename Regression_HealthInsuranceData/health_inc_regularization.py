import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

plt.close('all')

#set color palette
color_palette=sns.xkcd_palette(['cherry red' , 'royal blue' , 'emerald' , 'barney purple', 'yellow orange', 'rust brown'])

#import data
dataset = pd.read_csv('F:\\Kaggle\\health insurance dataset\\insurance.csv')

#convert categorical variables to numerical
dataset_modified = pd.get_dummies(dataset, columns=['smoker', 'region', 'sex'], prefix=['smoker', 'region', 'sex'])
dataset_modified= dataset_modified.drop(['smoker_no'] , axis=1)
dataset_modified= dataset_modified.drop(['sex_male'] , axis=1)


from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#function performs repeated cross-validation
def repeated_cv_rmse(X_original, y_original , model, n_reps, n_splits):
    random.seed(1)

    # perform repeated cross validation
    rmse_scores = np.zeros(n_reps)
    r2_scores = np.zeros(n_reps)

    for u in range(n_reps):

        #randomly shuffle the dataset
        indices = np.arange(X_original.shape[0])
        np.random.shuffle(indices)
        X = X_original[indices]
        y = y_original[indices] #dataset has been randomly shuffled

        #initialize vector to keep predictions from all folds of the cross-validation
        y_predicted = np.zeros(y.shape)

        #perform 10-fold cross validation
        kf = KFold(n_splits=n_splits , random_state=142)
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
            reg = model
            reg.fit(X_train, y_train)

            #make predictions on the testing set
            y_predicted[test] = reg.predict(X_test)

        #calculate rmse
        rmse_scores[u] =  np.sqrt(mean_squared_error(y, y_predicted))

    return rmse_scores


###Try Ridge Regression when there is noisy features
from sklearn import linear_model

n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values
X2= np.concatenate([X1, np.random.randn(X1.shape[0],500)] , axis= 1)
X3= np.concatenate([X1, np.random.randn(X1.shape[0],50)] , axis= 1)
y= dataset_modified['charges'].values

#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1 = repeated_cv_rmse(X1, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector2 = repeated_cv_rmse(X2, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector3 = repeated_cv_rmse(X3, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))


#plot parameter vs. error
fig = plt.subplot();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:green' , linewidth=4)
plt.xlabel('alpha')
plt.ylabel('RMSE ($)')
plt.title('RMSE vs. Hyper-parameter Alpha Ridge')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()








