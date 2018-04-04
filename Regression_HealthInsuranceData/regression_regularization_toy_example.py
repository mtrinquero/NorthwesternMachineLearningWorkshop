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


from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#function performs repeated cross-validation
def repeated_cv_rmse(X_original, y_original , model, n_reps, n_splits):
    random.seed(1)

    # perform repeated cross validation
    rmse_scores = np.zeros(n_reps)
    r2_scores = np.zeros(n_reps)
    coefs = np.zeros(shape=[n_reps , X_original.shape[1]])

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

        #train final model and get coefficients
        scaler = preprocessing.StandardScaler().fit(X_original)
        X_original_scaled = scaler.transform(X_original)
        reg= model
        reg.fit(X_original_scaled , y_original)
        coefs[u, :] = reg.coef_


        #calculate rmse
        rmse_scores[u] =  np.sqrt(mean_squared_error(y, y_predicted))

    return rmse_scores, coefs






###############################################################
#######################Ridge Regression########################
###############################################################
##Ridge regression with irrelevant features
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

n_reps = 10
n_splits = 5
X1 = np.random.randn(1000, 3) #red
X2= np.concatenate([X1, np.random.randn(X1.shape[0],5)] , axis= 1) #green
X3= np.concatenate([X1, np.random.randn(X1.shape[0],25)] , axis= 1) #blue
X4= np.concatenate([X1, np.random.randn(X1.shape[0],100)] , axis= 1) #purple
y = X1[:,0] + X1[:,1] + X1[:,2] + np.random.randn(1000)


#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
rmse_sweep4 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
std_parameter_sweep4 = []
coeff0_mean_sweep1 = []
coeff0_mean_sweep2 = []
coeff0_mean_sweep3 = []
coeff0_mean_sweep4 = []
coeff10_mean = []
coeff0_std_sweep1 = []
coeff0_std_sweep2 = []
coeff0_std_sweep3 = []
coeff0_std_sweep4 = []
coeff10_std = []

for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1, coef1 = repeated_cv_rmse(X1, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector2, coef2 = repeated_cv_rmse(X2, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector3, coef3 = repeated_cv_rmse(X3, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector4, coef4 = repeated_cv_rmse(X4, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))

    #append coefficients
    coeff0_mean_sweep1.append(np.mean(coef1[:,0]))
    coeff0_std_sweep1.append(np.std(coef1[:,0]))
    coeff0_mean_sweep2.append(np.mean(coef2[:,0]))
    coeff0_std_sweep2.append(np.std(coef2[:,0]))
    coeff0_mean_sweep3.append(np.mean(coef3[:,0]))
    coeff0_std_sweep3.append(np.std(coef3[:,0]))
    coeff0_mean_sweep4.append(np.mean(coef4[:,0]))
    coeff0_std_sweep4.append(np.std(coef4[:,0]))
    coeff10_mean.append(np.mean(coef4[:,9]))
    coeff10_std.append(np.std(coef4[:,9]))

#plot parameter vs. error
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs. Hyper-parameter Alpha, Ridge Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()


#plot parameter vs. coefficients
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep1) - np.array(coeff0_std_sweep1) ,
                 np.array(coeff0_mean_sweep1) + np.array(coeff0_std_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep2) - np.array(coeff0_std_sweep2) ,
                 np.array(coeff0_mean_sweep2) + np.array(coeff0_std_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep3) - np.array(coeff0_std_sweep3) ,
                 np.array(coeff0_mean_sweep3) + np.array(coeff0_std_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep4) - np.array(coeff0_std_sweep4) ,
                 np.array(coeff0_mean_sweep4) + np.array(coeff0_std_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff10_mean) - np.array(coeff10_std) ,
                 np.array(coeff10_mean) + np.array(coeff10_std) , facecolor = 'xkcd:light orange', alpha=0.5)
plt.semilogx(alpha_sweep,coeff10_mean , color= 'xkcd:orange' , linewidth=4)

plt.xlabel('alpha')
plt.ylabel('Coefficient')
plt.title('Coefficient vs. Hyper-parameter Alpha, Ridge Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()




###############################################################
#######################Lasso Regression########################
###############################################################
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

n_reps = 10
n_splits = 5
X1 = np.random.randn(1000, 3) #red
X2= np.concatenate([X1, np.random.randn(X1.shape[0],5)] , axis= 1) #green
X3= np.concatenate([X1, np.random.randn(X1.shape[0],25)] , axis= 1) #blue
X4= np.concatenate([X1, np.random.randn(X1.shape[0],100)] , axis= 1) #purple
y = X1[:,0] + X1[:,1] + X1[:,2] + np.random.randn(1000)


#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
rmse_sweep4 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
std_parameter_sweep4 = []
coeff0_mean_sweep1 = []
coeff0_mean_sweep2 = []
coeff0_mean_sweep3 = []
coeff0_mean_sweep4 = []
coeff10_mean = []
coeff0_std_sweep1 = []
coeff0_std_sweep2 = []
coeff0_std_sweep3 = []
coeff0_std_sweep4 = []
coeff10_std = []

for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1, coef1 = repeated_cv_rmse(X1, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)
    rmse_vector2, coef2 = repeated_cv_rmse(X2, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)
    rmse_vector3, coef3 = repeated_cv_rmse(X3, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)
    rmse_vector4, coef4 = repeated_cv_rmse(X4, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))

    #append coefficients
    coeff0_mean_sweep1.append(np.mean(coef1[:,0]))
    coeff0_std_sweep1.append(np.std(coef1[:,0]))
    coeff0_mean_sweep2.append(np.mean(coef2[:,0]))
    coeff0_std_sweep2.append(np.std(coef2[:,0]))
    coeff0_mean_sweep3.append(np.mean(coef3[:,0]))
    coeff0_std_sweep3.append(np.std(coef3[:,0]))
    coeff0_mean_sweep4.append(np.mean(coef4[:,0]))
    coeff0_std_sweep4.append(np.std(coef4[:,0]))
    coeff10_mean.append(np.mean(coef4[:,9]))
    coeff10_std.append(np.std(coef4[:,9]))

#plot parameter vs. error
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs. Hyper-parameter Alpha, Lasso Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()


#plot parameter vs. coefficients
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep1) - np.array(coeff0_std_sweep1) ,
                 np.array(coeff0_mean_sweep1) + np.array(coeff0_std_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep2) - np.array(coeff0_std_sweep2) ,
                 np.array(coeff0_mean_sweep2) + np.array(coeff0_std_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep3) - np.array(coeff0_std_sweep3) ,
                 np.array(coeff0_mean_sweep3) + np.array(coeff0_std_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep4) - np.array(coeff0_std_sweep4) ,
                 np.array(coeff0_mean_sweep4) + np.array(coeff0_std_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff10_mean) - np.array(coeff10_std) ,
                 np.array(coeff10_mean) + np.array(coeff10_std) , facecolor = 'xkcd:light orange', alpha=0.5)
plt.semilogx(alpha_sweep,coeff10_mean , color= 'xkcd:orange' , linewidth=4)

plt.xlabel('alpha')
plt.ylabel('Coefficient')
plt.title('Coefficient vs. Hyper-parameter Alpha, Lasso Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()



###############################################################
#######################Ridge Regression Nonlinear#
###############################################################
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

n_reps = 10
n_splits = 5
X1 = np.random.randn(1000, 3) #red
X2= np.concatenate([X1, np.power(X1[:,0],2).reshape(-1,1)] , axis= 1) #green
X3= np.concatenate([X1, np.power(X1,2) ] , axis= 1) #blue
X4= np.concatenate([X1, np.power(X1,2) , np.power(X1,3), np.power(X1,4), np.power(X1,5) , np.power(X1,6),  np.power(X1,7)  ], axis= 1) #purple
y = np.power(X1[:,0],2)  +X1[:,1] + X1[:,2] + np.random.randn(1000)


#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
rmse_sweep4 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
std_parameter_sweep4 = []
coeff0_mean_sweep1 = []
coeff0_mean_sweep2 = []
coeff0_mean_sweep3 = []
coeff0_mean_sweep4 = []
coeff10_mean = []
coeff0_std_sweep1 = []
coeff0_std_sweep2 = []
coeff0_std_sweep3 = []
coeff0_std_sweep4 = []
coeff10_std = []

for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1, coef1 = repeated_cv_rmse(X1, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector2, coef2 = repeated_cv_rmse(X2, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector3, coef3 = repeated_cv_rmse(X3, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector4, coef4 = repeated_cv_rmse(X4, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))

    #append coefficients
    coeff0_mean_sweep1.append(np.mean(coef1[:,0]))
    coeff0_std_sweep1.append(np.std(coef1[:,0]))
    coeff0_mean_sweep2.append(np.mean(coef2[:,0]))
    coeff0_std_sweep2.append(np.std(coef2[:,0]))
    coeff0_mean_sweep3.append(np.mean(coef3[:,0]))
    coeff0_std_sweep3.append(np.std(coef3[:,0]))
    coeff0_mean_sweep4.append(np.mean(coef4[:,0]))
    coeff0_std_sweep4.append(np.std(coef4[:,0]))
    coeff10_mean.append(np.mean(coef4[:,9]))
    coeff10_std.append(np.std(coef4[:,9]))

#plot parameter vs. error
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs. Hyper-parameter Alpha, Ridge Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()






###############################################################
#######################Lasso Regression Nonlinear#
###############################################################
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

n_reps = 10
n_splits = 5
X1 = np.random.randn(1000, 3) #red
X2= np.concatenate([X1, np.power(X1[:,0],2).reshape(-1,1)] , axis= 1) #green
X3= np.concatenate([X1, np.power(X1,2) ] , axis= 1) #blue
X4= np.concatenate([X1, np.power(X1,2) , np.power(X1,3), np.power(X1,4), np.power(X1,5) , np.power(X1,6),  np.power(X1,7)  ], axis= 1) #purple
y = np.power(X1[:,0],2)  +X1[:,1] + X1[:,2] + np.random.randn(1000)


#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
rmse_sweep4 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
std_parameter_sweep4 = []
coeff0_mean_sweep1 = []
coeff0_mean_sweep2 = []
coeff0_mean_sweep3 = []
coeff0_mean_sweep4 = []
coeff10_mean = []
coeff0_std_sweep1 = []
coeff0_std_sweep2 = []
coeff0_std_sweep3 = []
coeff0_std_sweep4 = []
coeff10_std = []

for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1, coef1 = repeated_cv_rmse(X1, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)
    rmse_vector2, coef2 = repeated_cv_rmse(X2, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)
    rmse_vector3, coef3 = repeated_cv_rmse(X3, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)
    rmse_vector4, coef4 = repeated_cv_rmse(X4, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))

    #append coefficients
    coeff0_mean_sweep1.append(np.mean(coef1[:,0]))
    coeff0_std_sweep1.append(np.std(coef1[:,0]))
    coeff0_mean_sweep2.append(np.mean(coef2[:,0]))
    coeff0_std_sweep2.append(np.std(coef2[:,0]))
    coeff0_mean_sweep3.append(np.mean(coef3[:,0]))
    coeff0_std_sweep3.append(np.std(coef3[:,0]))
    coeff0_mean_sweep4.append(np.mean(coef4[:,0]))
    coeff0_std_sweep4.append(np.std(coef4[:,0]))
    coeff10_mean.append(np.mean(coef4[:,9]))
    coeff10_std.append(np.std(coef4[:,9]))

#plot parameter vs. error
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs. Hyper-parameter Alpha, Lasso Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()








###############################################################
#######################Ridge Regression Colinear#
###############################################################
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

n_reps = 10
n_splits = 5
X1 = np.random.randn(1000, 3) #red
X2= np.concatenate([X1, X1[:,0].reshape(-1,1) ] , axis= 1) #green
X3= np.concatenate([X1, X1+ 0.00005*np.random.randn(1000, 3)] , axis= 1) #blue
X4= np.concatenate([X1, X1  ], axis= 1) #purple, orange
y = X1[:,0] + X1[:,1] + X1[:,2] + np.random.randn(1000)


#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
rmse_sweep4 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
std_parameter_sweep4 = []
coeff0_mean_sweep1 = []
coeff0_mean_sweep2 = []
coeff0_mean_sweep3 = []
coeff0_mean_sweep4 = []
coeff10_mean = []
coeff0_std_sweep1 = []
coeff0_std_sweep2 = []
coeff0_std_sweep3 = []
coeff0_std_sweep4 = []
coeff10_std = []

for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1, coef1 = repeated_cv_rmse(X1, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector2, coef2 = repeated_cv_rmse(X2, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector3, coef3 = repeated_cv_rmse(X3, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)
    rmse_vector4, coef4 = repeated_cv_rmse(X4, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))

    #append coefficients
    coeff0_mean_sweep1.append(np.mean(coef1[:,0]))
    coeff0_std_sweep1.append(np.std(coef1[:,0]))
    coeff0_mean_sweep2.append(np.mean(coef2[:,0]))
    coeff0_std_sweep2.append(np.std(coef2[:,0]))
    coeff0_mean_sweep3.append(np.mean(coef3[:,0]))
    coeff0_std_sweep3.append(np.std(coef3[:,0]))
    coeff0_mean_sweep4.append(np.mean(coef4[:,0]))
    coeff0_std_sweep4.append(np.std(coef4[:,0]))
    coeff10_mean.append(np.mean(coef4[:,3]))
    coeff10_std.append(np.std(coef4[:,3]))


#plot parameter vs. error
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs. Hyper-parameter Alpha, Ridge Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()


#plot parameter vs. coefficients
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep1) - np.array(coeff0_std_sweep1) ,
                 np.array(coeff0_mean_sweep1) + np.array(coeff0_std_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep1 , color= 'xkcd:red' , linewidth=8)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep2) - np.array(coeff0_std_sweep2) ,
                 np.array(coeff0_mean_sweep2) + np.array(coeff0_std_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep2 , color= 'xkcd:green' , linewidth=6)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep3) - np.array(coeff0_std_sweep3) ,
                 np.array(coeff0_mean_sweep3) + np.array(coeff0_std_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep3 , color= 'xkcd:blue' , linewidth=2)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff0_mean_sweep4) - np.array(coeff0_std_sweep4) ,
                 np.array(coeff0_mean_sweep4) + np.array(coeff0_std_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,coeff0_mean_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
#plot parameter vs. coefficients
plt.fill_between(alpha_sweep , np.array(coeff10_mean) - np.array(coeff10_std) ,
                 np.array(coeff10_mean) + np.array(coeff10_std) , facecolor = 'xkcd:light orange', alpha=0.5)
plt.semilogx(alpha_sweep,coeff10_mean , color= 'xkcd:orange' , linewidth=1)

plt.xlabel('alpha')
plt.ylabel('Coefficient')
plt.title('Coefficient vs. Hyper-parameter Alpha, Ridge Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()



