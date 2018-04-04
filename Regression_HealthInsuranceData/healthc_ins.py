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

plt.close('all')

#set color palette
color_palette=sns.xkcd_palette(['cherry red' , 'royal blue' , 'emerald' , 'barney purple', 'yellow orange', 'rust brown'])

#import data
dataset = pd.read_csv('F:\\Kaggle\\health insurance dataset\\insurance.csv')

#look at dataset
print(dataset.head())

#how does age relate to charges
g = sns.lmplot(x='age', y='charges', data=dataset)
g = sns.lmplot(x='age', y='charges', hue = 'sex' , data=dataset , palette=color_palette )

#... explore other variables as well ...
g = sns.lmplot(x='age', y='charges', hue = 'children' , data=dataset , palette=color_palette)
g = sns.lmplot(x='age', y='charges', hue = 'smoker' , data=dataset , palette=color_palette) ##
g = sns.lmplot(x='age', y='charges', hue = 'region' , data=dataset , palette=color_palette)

#how does bmi relate to charges
g = sns.lmplot(x='bmi', y='charges', data=dataset , palette=color_palette)
g = sns.lmplot(x='bmi', y='charges', hue = 'sex' , data=dataset , palette=color_palette)

#... explore other variables as well ...
g = sns.lmplot(x='bmi', y='charges', hue = 'children' , data=dataset , palette=color_palette)
g = sns.lmplot(x='bmi', y='charges', hue = 'smoker' , data=dataset, palette=color_palette)#
g = sns.lmplot(x='bmi', y='charges', hue = 'region' , data=dataset, palette=color_palette)


from sklearn import linear_model

#look at slope for interpretation
regr_none_smoker_bmi = linear_model.LinearRegression()
regr_none_smoker_bmi.fit(dataset[ dataset['smoker'] == 'no' ]['bmi'].reshape(-1,1) , dataset[dataset['smoker'] == 'no']['charges'].reshape(-1,1) )
print(regr_none_smoker_bmi.coef_)

regr_smoker_age = linear_model.LinearRegression()
regr_smoker_age.fit(dataset[ dataset['smoker'] == 'yes' ]['bmi'].reshape(-1,1) , dataset[dataset['smoker'] == 'yes']['charges'].reshape(-1,1) )
print(regr_smoker_age.coef_)

##...interpretation of coefficients

#look at categorical varialbes
plt.figure()
sns.barplot(x='sex', y='charges' , data=dataset , errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='sex', y='charges', hue ='smoker' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)

##...draw and interpret other features
plt.figure()
sns.barplot(x='sex', y='charges', hue ='children' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='sex', y='charges', hue ='region' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='children', y='charges', data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='children', y='charges', hue ='sex' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='children', y='charges', hue ='smoker' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='children', y='charges', hue ='region' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='region', y='charges', data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='region', y='charges', hue ='sex' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='region', y='charges', hue ='smoker' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='region', y='charges', hue ='children' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='smoker', y='charges', data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='smoker', y='charges', hue ='sex' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='smoker', y='charges', hue ='region' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)
plt.figure()
sns.barplot(x='smoker', y='charges', hue ='children' , data=dataset, errcolor ='.1', edgecolor='.1', alpha=0.8, capsize=0.05, palette=color_palette)


#... how to deal with categorical variables ??
print(dataset.head())

#dealing with categorical variables
dataset_scatterplot = dataset
dataset_scatterplot['smoker']= dataset_scatterplot['smoker'].astype('category')
dataset_scatterplot['smoker'] = dataset_scatterplot['smoker'].cat.codes
print(dataset_scatterplot.head())

#deal with rest of the categorical variables
dataset_scatterplot['region']= dataset_scatterplot['region'].astype('category')
dataset_scatterplot['sex']= dataset_scatterplot['sex'].astype('category')
dataset_scatterplot['region'] = dataset_scatterplot['region'].cat.codes
dataset_scatterplot['sex'] = dataset_scatterplot['sex'].cat.codes
print(dataset_scatterplot.head())

#alternative way of visualizing data: scatter plots
g = sns.pairplot(dataset_scatterplot, diag_kind="kde", kind ='reg', hue='smoker', markers="+",  diag_kws=dict(shade=True), palette=color_palette)
g = sns.pairplot(dataset_scatterplot, diag_kind="kde", kind ='reg', hue='children', markers="+",  diag_kws=dict(shade=True), palette=color_palette)
g = sns.pairplot(dataset_scatterplot, diag_kind="kde", kind ='reg', hue='region', markers="+",  diag_kws=dict(shade=True), palette=color_palette)
g = sns.pairplot(dataset_scatterplot, diag_kind="kde", kind ='reg', hue='sex', markers="+",  diag_kws=dict(shade=True), palette=color_palette)


#one-hot encoding
#https://chrisalbon.com/machine_learning/preprocessing_structured_data/one-hot_encode_nominal_categorical_features/
dataset = pd.read_csv('F:\\Kaggle\\health insurance dataset\\insurance.csv')
print(dataset.head())
dataset_modified = pd.get_dummies(dataset, columns=['smoker', 'region', 'sex'], prefix=['smoker', 'region', 'sex'])
print(dataset_modified.head())
dataset_modified= dataset_modified.drop(['smoker_no'] , axis=1)
dataset_modified= dataset_modified.drop(['sex_male'] , axis=1)
print(dataset_modified.head())

#pearson correlation
#https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#/media/File:Correlation_examples2.svg
rho=np.corrcoef(dataset_modified.values, rowvar=False)
list_groups=dataset_modified.columns

fig = plt.figure();
sns.set_style('whitegrid', {'grid.linestyle':'--'})
mask=np.zeros_like(rho)
mask[np.triu_indices_from(mask)]=True
sns.set(font_scale=1)
sns.heatmap(rho , mask = mask , xticklabels=list_groups , yticklabels=list_groups, linewidth = 1, cmap = 'seismic', vmin = -1, vmax = 1 )
plt.tight_layout()
plt.show()


from scipy import stats

#spearman correlation
#https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
rho,pval=stats.spearmanr(dataset_modified.values)
list_groups=dataset_modified.columns

fig = plt.figure();
sns.set_style('whitegrid', {'grid.linestyle':'--'})
mask=np.zeros_like(rho)
mask[np.triu_indices_from(mask)]=True
sns.set(font_scale=1)
sns.heatmap(rho , mask = mask , xticklabels=list_groups , yticklabels=list_groups, linewidth = 1, cmap = 'seismic', vmin = -1, vmax = 1 )
plt.tight_layout()
plt.show()

fig = plt.figure();
sns.set_style('whitegrid', {'grid.linestyle':'--'})
mask=np.zeros_like(rho)
mask[np.triu_indices_from(mask)]=True
sns.set(font_scale=1)
sns.heatmap((pval<0.05).astype(int) , mask = mask , xticklabels=list_groups , yticklabels=list_groups, linewidth = 1, cmap = 'Reds', vmin =0 , vmax = 1 )
plt.tight_layout()
plt.show()


#TSNE plots
from sklearn import manifold
from sklearn import preprocessing

#transform data using tsne
tsne = manifold.TSNE(n_components=2, init='pca' , method='barnes_hut' , random_state=4 , perplexity=60) #initialize tsne
list_of_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']
X= dataset_modified[list_of_features  ].values
y= dataset_modified['charges'].values
X_standardized = preprocessing.scale(X) #standardize your data
X_tsne= tsne.fit_transform(X_standardized) #perform tsne

fig = plt.subplots()
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=np.log10(y), s= 30, cmap= 'jet', alpha = 0.9 , edgecolors='none')
plt.title('Charges')

fig = plt.subplots()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dataset_modified[ 'smoker_yes'], s=30, cmap='jet', alpha=0.9, edgecolors='none')
plt.title('smoker_yes')

#... look at other features with t-SNE
#... try different feature sets

for u in list_of_features:
    fig = plt.subplots()
    plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=dataset_modified[u], s= 30, cmap= 'jet', alpha = 0.9 , edgecolors='none')
    plt.title(u)

fig = plt.subplots()
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=np.log10(y), s= 30, cmap= 'jet', alpha = 0.9 , edgecolors='none')
plt.title('Charges')

###lasso and ridge regression
#http://businessforecastblog.com/wp-content/uploads/2014/05/LASSOobkf.png
list_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']
X1= dataset_modified[list_features].values
y= dataset_modified['charges'].values
reg= linear_model.Lasso(alpha=1)
reg.fit(X1,y)
print(list_features)
print(reg.coef_)
plt.scatter(list_features,reg.coef_)
#... we should have standardize ...

reg.fit(preprocessing.scale(X1),y)
print(reg.coef_)
plt.figure()
plt.scatter(list_features,reg.coef_)

#test myself
scaler = preprocessing.StandardScaler().fit(X1)
x_test = scaler.transform(np.array([26, 21, 0, 0 ,0 , 0 , 1, 0 , 0]).reshape(1,-1))
print(reg.predict(x_test))

##Function for repeated cross validation
from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import preprocessing

#function performs repeated cross-validation
def repeated_cv_rmse(X_original, y_original , model, n_reps, n_splits, standardize):
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
            if(standardize):
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



###############################################################
#######################Ridge Regression########################
###############################################################

#https://chrisalbon.com/machine_learning/linear_regression/ridge_regression/

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values #red
X2= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'sex_female']].values #green
X3= dataset_modified[['age', 'bmi', 'children', 'smoker_yes']].values #blue
X4= dataset_modified[['age', 'bmi', 'children']].values #purple
y= dataset_modified['charges'].values


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
for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1 = repeated_cv_rmse(X1, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits, True)
    rmse_vector2 = repeated_cv_rmse(X2, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits, True)
    rmse_vector3 = repeated_cv_rmse(X3, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits, True)
    rmse_vector4 = repeated_cv_rmse(X4, y, linear_model.Ridge(alpha=alpha), n_reps, n_splits, True)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))


#plot parameter vs. error
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=5)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=3)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=2)
plt.xlabel('alpha')
plt.ylabel('RMSE ($)')
plt.title('RMSE vs. Hyper-parameter Alpha, Ridge Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()



###############################################################
#######################Lasso Regression########################
###############################################################
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values #red
X2= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'sex_female']].values #green
X3= dataset_modified[['age', 'bmi', 'children', 'smoker_yes']].values #blue
X4= dataset_modified[['age', 'bmi', 'children']].values #purple
y= dataset_modified['charges'].values


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
for alpha in alpha_sweep:

    #perform repeated cross-validation
    rmse_vector1 = repeated_cv_rmse(X1, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits, True)
    rmse_vector2 = repeated_cv_rmse(X2, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits, True)
    rmse_vector3 = repeated_cv_rmse(X3, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits, True)
    rmse_vector4 = repeated_cv_rmse(X4, y, linear_model.Lasso(alpha=alpha), n_reps, n_splits, True)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))


#plot parameter vs. error
fig = plt.figure();
plt.fill_between(alpha_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=5)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=3)
#plot parameter vs. error
plt.fill_between(alpha_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.semilogx(alpha_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=2)
plt.xlabel('alpha')
plt.ylabel('RMSE ($)')
plt.title('RMSE vs. Hyper-parameter Alpha, Lasso Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()




#looking into the coefficients in Lasso
from sklearn import linear_model

list_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']
X1= dataset_modified[list_features].values
y= dataset_modified['charges'].values

#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#initialize array to keep coeffs
coeffs = np.zeros([X1.shape[1],alpha_sweep.shape[0]])

count = 0
for alpha in alpha_sweep:

    #perform repeated cross-validation
    scaler = preprocessing.StandardScaler().fit(X1)
    X_original_scaled = scaler.transform(X1)
    reg= linear_model.Lasso(alpha=alpha)
    reg.fit(X_original_scaled , y)
    coeffs[:,count] = reg.coef_
    count = count+1

#plot parameter vs. error
fig=plt.figure()
ax  = plt.subplot(1,1,1);
ax.semilogx(alpha_sweep,coeffs[0,:] , linewidth=4 , label = list_features[0])
ax.semilogx(alpha_sweep,coeffs[1,:] , linewidth=4 , label = list_features[1])
ax.semilogx(alpha_sweep,coeffs[2,:] , linewidth=4 , label = list_features[2])
ax.semilogx(alpha_sweep,coeffs[3,:] , linewidth=4 , label = list_features[3])
ax.semilogx(alpha_sweep,coeffs[4,:] , linewidth=4 , label = list_features[4], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[5,:] , linewidth=4 , label = list_features[5], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[6,:] , linewidth=4 , label = list_features[6], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[7,:] , linewidth=4 , label = list_features[7], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[8,:] , linewidth=4 , label = list_features[8], linestyle = '--')
ax.legend()
plt.xlabel('alpha')
plt.ylabel('Coefficient')
plt.title('Coefficient vs. Hyper-parameter Alpha Lasso')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()







#looking into the coefficients in Ridge
from sklearn import linear_model

list_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']
X1= dataset_modified[list_features].values
y= dataset_modified['charges'].values

#set up the parameter sweep
alpha_sweep =  np.power(10, np.linspace(-4,4,100))

#initialize array to keep coeffs
coeffs = np.zeros([X1.shape[1],alpha_sweep.shape[0]])

count = 0
for alpha in alpha_sweep:

    #perform repeated cross-validation
    scaler = preprocessing.StandardScaler().fit(X1)
    X_original_scaled = scaler.transform(X1)
    reg= linear_model.Ridge(alpha=alpha)
    reg.fit(X_original_scaled , y)
    coeffs[:,count] = reg.coef_
    count = count+1

#plot parameter vs. error
fig=plt.figure()
ax  = plt.subplot(1,1,1);
ax.semilogx(alpha_sweep,coeffs[0,:] , linewidth=4 , label = list_features[0])
ax.semilogx(alpha_sweep,coeffs[1,:] , linewidth=4 , label = list_features[1])
ax.semilogx(alpha_sweep,coeffs[2,:] , linewidth=4 , label = list_features[2])
ax.semilogx(alpha_sweep,coeffs[3,:] , linewidth=4 , label = list_features[3])
ax.semilogx(alpha_sweep,coeffs[4,:] , linewidth=4 , label = list_features[4], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[5,:] , linewidth=4 , label = list_features[5], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[6,:] , linewidth=4 , label = list_features[6], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[7,:] , linewidth=4 , label = list_features[7], linestyle = '--')
ax.semilogx(alpha_sweep,coeffs[8,:] , linewidth=4 , label = list_features[8], linestyle = '--')
ax.legend()
plt.xlabel('alpha')
plt.ylabel('Coefficient')
plt.title('Coefficient vs. Hyper-parameter Alpha Ridge')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()



####DIVERGE HERE TO TALK MORE ABOUT REGULARIZATION
##health_inc_regularization.py
##regression_regularization_toy_example.py




#############Intro to regression trees
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


from sklearn.tree import DecisionTreeRegressor

list_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']
X1= dataset_modified[list_features].values
y= dataset_modified['charges'].values

#set up regression tree
#regressor = DecisionTreeRegressor(random_state=0, max_depth=3)
#regressor = DecisionTreeRegressor(random_state=0, max_depth=6 , min_samples_split=0.1)
#regressor = DecisionTreeRegressor(random_state=0, max_depth=6 , min_samples_split=0.1,  min_samples_leaf = 0.1)
regressor = DecisionTreeRegressor(random_state=0, max_depth=16 , min_samples_split=0.01,  min_samples_leaf = 0.2)
regressor.fit(X1, y)

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydot
#install graphviz than pydot

dot_data = StringIO()
dot_data= export_graphviz(regressor, out_file='tree.dot')
temp=export_graphviz(regressor,
        out_file=dot_data,
        feature_names=list_features,
        impurity=False,
        filled=True, rounded=True)
graph = pydot.graph_from_dot_data(temp)
graph[0].write_pdf("tree_model.pdf")






#demonstrate
x_test = np.array([26, 21, 0, 0 ,0 , 0 , 1, 0 , 0])
print(regressor.predict(x_test))

#plot feature importances
fig = plt.figure(figsize=(18,10))
plt.scatter(list_features, regressor.feature_importances_ , s =200*regressor.feature_importances_+75 , c = regressor.feature_importances_  , cmap = 'jet')
plt.title('Feature Importances For Regression Tree')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.tight_layout()


###############################################################
#######################Tune Tree 1D########################
###############################################################
from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values #red
y= dataset_modified['charges'].values

#set up the parameter sweep
min_samples_split_sweep =   np.linspace(0.01 , 0.9, 200)

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
rmse_sweep4 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
std_parameter_sweep4 = []

for u in min_samples_split_sweep:

    #perform repeated cross-validation
    rmse_vector1 = repeated_cv_rmse(X1, y, DecisionTreeRegressor(random_state=0, max_depth=16 , min_samples_split=u,  min_samples_leaf = 0.01), n_reps, n_splits, False)
    rmse_vector2 = repeated_cv_rmse(X1, y, DecisionTreeRegressor(random_state=0, max_depth=16 , min_samples_split=u,  min_samples_leaf = 0.1), n_reps, n_splits, False)
    rmse_vector3 = repeated_cv_rmse(X1, y, DecisionTreeRegressor(random_state=0, max_depth=16 , min_samples_split=u,  min_samples_leaf = 0.2), n_reps, n_splits, False)
    rmse_vector4 = repeated_cv_rmse(X1, y, DecisionTreeRegressor(random_state=0, max_depth=16 , min_samples_split=u,  min_samples_leaf = 0.3), n_reps, n_splits, False)

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))


#plot parameter vs. error
fig = plt.figure();
plt.fill_between(min_samples_split_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.plot(min_samples_split_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=4)
#plot parameter vs. error
plt.fill_between(min_samples_split_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.plot(min_samples_split_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(min_samples_split_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.plot(min_samples_split_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=4)
#plot parameter vs. error
plt.fill_between(min_samples_split_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.plot(min_samples_split_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=4)
plt.xlabel('min_samples_split')
plt.ylabel('RMSE ($)')
plt.title('RMSE vs. Hyper-parameter min_samples_split, Tree Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()



###############################################################
#######################Tune Tree 2D########################
###############################################################
from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values #red
y= dataset_modified['charges'].values

#set up the parameter sweep
n_sweep = 10
min_samples_split_range = np.linspace(0.01 , 0.9, n_sweep)
min_samples_leaf_range = np.linspace(0.01, 0.4, n_sweep)
min_samples_split_sweep, min_samples_leaf_sweep = np.meshgrid(min_samples_split_range, min_samples_leaf_range)

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = np.zeros(min_samples_split_sweep.shape) # keep scores here

for u in np.arange(n_sweep):
    for v in np.arange(n_sweep):

        #perform repeated cross-validation
        rmse_vector1 = repeated_cv_rmse(X1, y, DecisionTreeRegressor(random_state=0, max_depth=16 , min_samples_split=min_samples_split_sweep[u,v],
                                                                     min_samples_leaf =min_samples_leaf_sweep[u,v] ), n_reps, n_splits, False)


        ##append scores
        rmse_sweep1[u,v] = np.mean(rmse_vector1)


#plot parameter vs. error
plt.figure(figsize=(8, 6))
plt.imshow(rmse_sweep1, interpolation='nearest', cmap=plt.cm.jet)
plt.xlabel('min_samples_split')
plt.ylabel('min_samples_leaf')
plt.colorbar()
plt.xticks(np.arange(len(min_samples_split_range)), min_samples_split_range, rotation=45 )
plt.yticks(np.arange(len(min_samples_leaf_range)), min_samples_leaf_range)
plt.title('Validation accuracy')
plt.tight_layout()
plt.show()

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(min_samples_split_sweep.reshape(-1), min_samples_leaf_sweep.reshape(-1), rmse_sweep1.reshape(-1) ,cmap='jet')
ax.set_xlabel('min_samples_split')
ax.set_ylabel('min_samples_leaf')
ax.set_zlabel('RMSE ($)');




#######Use random forest regression
from sklearn.ensemble import RandomForestRegressor
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

plt.close('all')

#import data
dataset = pd.read_csv('/media/polatalemdar/9E82D1BB82D197DB/Kaggle/health insurance dataset/insurance.csv')

#convert categorical variables to numerical
dataset_modified = pd.get_dummies(dataset, columns=['smoker', 'region', 'sex'], prefix=['smoker', 'region', 'sex'])
dataset_modified= dataset_modified.drop(['smoker_no'] , axis=1)
dataset_modified= dataset_modified.drop(['sex_male'] , axis=1)

list_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']
X1= dataset_modified[list_features].values
y= dataset_modified['charges'].values

#set up regression tree
#regressor = RandomForestRegressor(n_estimators = 5, random_state=0)
#regressor = RandomForestRegressor(n_estimators = 5, max_depth=6 , min_samples_split=0.1)
regressor = RandomForestRegressor(n_estimators = 5, max_depth=6 , min_samples_split=0.1, min_samples_leaf=0.1, max_features=0.5)
regressor.fit(X1, y)

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydot

for u in range(5):
    tree=regressor.estimators_[u]

    dot_data = StringIO()
    dot_data= export_graphviz(tree, out_file='tree.dot')
    temp=export_graphviz(tree,
            out_file=dot_data,
            feature_names=list_features,
            impurity=False,
            filled=True, rounded=True)
    graph = pydot.graph_from_dot_data(temp)
    graph[0].write_pdf("rf"+str(u)+".pdf")


#demonstrate
x_test = np.array([26, 21, 0, 0 ,0 , 0 , 1, 0 , 0])
print(regressor.predict(x_test))

#plot feature importances
fig = plt.figure(figsize=(18,10))
plt.scatter(list_features, regressor.feature_importances_ , s =200*regressor.feature_importances_+75 , c = regressor.feature_importances_  , cmap = 'jet')
plt.title('Feature Importances For Regression Tree')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.tight_layout()




###############################################################
#######################Tune RF, min_samples_split########################
###############################################################
from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values #red
y= dataset_modified['charges'].values

#set up the parameter sweep
parameter_sweep =   np.linspace(0.01, 0.95, 20)

#perform repeated cross-validation by sweeping the parameter
rmse_sweep1 = [] # keep scores here
rmse_sweep2 = []
rmse_sweep3 = []
rmse_sweep4 = []
std_parameter_sweep1 = []
std_parameter_sweep2 = []
std_parameter_sweep3 = []
std_parameter_sweep4 = []

for u in parameter_sweep:

    print(u)

    #perform repeated cross-validation
    rmse_vector1 = repeated_cv_rmse(X1, y, RandomForestRegressor(n_estimators = 10, max_depth=2 , min_samples_split=0.1, min_samples_leaf=0.1, max_features=u), n_reps, n_splits, False) #red
    rmse_vector2 = repeated_cv_rmse(X1, y, RandomForestRegressor(n_estimators = 10, max_depth=4 , min_samples_split=0.1, min_samples_leaf=0.1, max_features=u), n_reps, n_splits, False) #green
    rmse_vector3 = repeated_cv_rmse(X1, y, RandomForestRegressor(n_estimators = 10, max_depth=8 , min_samples_split=0.1, min_samples_leaf=0.1, max_features=u), n_reps, n_splits, False) #blue
    rmse_vector4 = repeated_cv_rmse(X1, y, RandomForestRegressor(n_estimators = 10, max_depth=16 , min_samples_split=0.1, min_samples_leaf=0.1, max_features=u), n_reps, n_splits, False) #purple

    ##append scores
    rmse_sweep1.append(np.mean(rmse_vector1))
    std_parameter_sweep1.append(np.std(rmse_vector1))
    rmse_sweep2.append(np.mean(rmse_vector2))
    std_parameter_sweep2.append(np.std(rmse_vector2))
    rmse_sweep3.append(np.mean(rmse_vector3))
    std_parameter_sweep3.append(np.std(rmse_vector3))
    rmse_sweep4.append(np.mean(rmse_vector4))
    std_parameter_sweep4.append(np.std(rmse_vector4))


#plot parameter vs. error
fig = plt.figure();
plt.fill_between(parameter_sweep , np.array(rmse_sweep1) - np.array(std_parameter_sweep1) ,
                 np.array(rmse_sweep1) + np.array(std_parameter_sweep1) , facecolor = 'xkcd:light pink', alpha=0.5)
plt.plot(parameter_sweep,rmse_sweep1 , color= 'xkcd:red' , linewidth=5)
#plot parameter vs. error
plt.fill_between(parameter_sweep , np.array(rmse_sweep2) - np.array(std_parameter_sweep2) ,
                 np.array(rmse_sweep2) + np.array(std_parameter_sweep2) , facecolor = 'xkcd:light green', alpha=0.5)
plt.plot(parameter_sweep,rmse_sweep2 , color= 'xkcd:green' , linewidth=4)
#plot parameter vs. error
plt.fill_between(parameter_sweep , np.array(rmse_sweep3) - np.array(std_parameter_sweep3) ,
                 np.array(rmse_sweep3) + np.array(std_parameter_sweep3) , facecolor = 'xkcd:light blue', alpha=0.5)
plt.plot(parameter_sweep,rmse_sweep3 , color= 'xkcd:blue' , linewidth=3)
#plot parameter vs. error
plt.fill_between(parameter_sweep , np.array(rmse_sweep4) - np.array(std_parameter_sweep4) ,
                 np.array(rmse_sweep4) + np.array(std_parameter_sweep4) , facecolor = 'xkcd:light purple', alpha=0.5)
plt.plot(parameter_sweep,rmse_sweep4 , color= 'xkcd:barney purple' , linewidth=2)
plt.xlabel('parameter')
plt.ylabel('RMSE ($)')
plt.title('RMSE vs. Hyper-parameter, RF Regression')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()



#demonstrating hyperopt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import math

#define objetive function to be minimized
def objective(params):
    x= params['x']
    y= params['y']
    return {'loss': abs(x ** 2 - y**2 + x*y) , 'status': STATUS_OK }

#define search space
space = { 'x' :  hp.uniform('x', -20,20) , 'y' :  hp.uniform('y', -20,20)}

#start searching
trials = Trials()
best = fmin(objective ,
    space= space,
    algo=tpe.suggest,
    max_evals=5000,
    trials = trials)

print('Bes results:')
print(best)

#visualize search
x_vals=[]
y_vals=[]
for u in range(len(trials.trials)):
    x_vals.append(trials.trials[u]['misc']['vals']['x'])
    y_vals.append(trials.trials[u]['misc']['vals']['y'])

fig= plt.figure(figsize = (12,8))
plt.subplot(121)
plt.plot(x_vals)
plt.title('x values in search')
plt.xlabel('iteration number')
plt.ylabel('x')

plt.subplot(122)
plt.plot(y_vals)
plt.title('y values in search')
plt.xlabel('iteration number')
plt.ylabel('y')


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x_vals, y_vals, trials.losses(), c=np.arange(len(x_vals)), cmap='jet')
ax.scatter(best['x'], best['y'], objective({'x': best['x'], 'y': best['y']} )['loss'] , marker='x', color='black' ,s=300 , linewidth = 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)');



###Tuning Regression Tree Using HyperOpt
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import fmin
from sklearn.tree import DecisionTreeRegressor


#convert categorical variables to numerical
dataset = pd.read_csv('/media/polatalemdar/9E82D1BB82D197DB/Kaggle/health insurance dataset/insurance.csv')
dataset_modified = pd.get_dummies(dataset, columns=['smoker', 'region', 'sex'], prefix=['smoker', 'region', 'sex'])
dataset_modified= dataset_modified.drop(['smoker_no'] , axis=1)
dataset_modified= dataset_modified.drop(['sex_male'] , axis=1)

n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values #red
y= dataset_modified['charges'].values
list_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']

#define objective function to minimize
def objective(space):



    model=DecisionTreeRegressor(random_state=0, max_depth=space['max_depth'] ,
                          min_samples_split=space['min_samples_split'],
                          min_samples_leaf = space['min_samples_leaf'],
                          max_features = space['max_features'],
                          criterion=space['criterion']
                          )
    score_vector = repeated_cv_rmse(X1, y , model, n_reps, n_splits, (space['standardize']==1) )
    score = np.mean(score_vector)
    print("RMSE {} space {}".format(score, space))
    return {'loss': score, 'status': STATUS_OK }

space = {
    'max_depth': hp.choice('max_depth', [2, 4 , 8 ,16, 32] ),
    'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.9),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.45),
    'max_features': hp.uniform('max_features', 0.1 , 1),
    'criterion': hp.choice('criterion', ['mse', 'friedman_mse', 'mae'] ),
    'standardize':  hp.choice('standardize', [0,1] ),
}


#start searching
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials)
print('Best Model:')
print(hyperopt.space_eval(space,best))

#visualize search
losses_list=trials.losses()
min_samples_split_list=[]
criterion_list = []
max_depth_list = []
max_features_list = []
min_samples_leaf_list = []

for u in range(len(trials.trials)):

    min_samples_split_list.append(trials.trials[u]['misc']['vals']['min_samples_split'])
    criterion_list.append(trials.trials[u]['misc']['vals']['criterion'])
    max_depth_list.append(trials.trials[u]['misc']['vals']['max_depth'])
    max_features_list.append(trials.trials[u]['misc']['vals']['max_features'])
    min_samples_leaf_list.append(trials.trials[u]['misc']['vals']['min_samples_leaf'])

#figure out which iteration the optimal tree was found
print('Optimal Tree Found in iteration number:')
index = losses_list.index(min(losses_list))
print(index)
print('Best Model RMSE:')
print(losses_list[index])

fig= plt.figure(figsize = (15,9))
plt.subplot(321)
plt.plot(losses_list)
plt.plot(index, losses_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('RMSE')

plt.subplot(322)
plt.plot(min_samples_split_list)
plt.plot(index, min_samples_split_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('min_samples_split')

plt.subplot(323)
plt.plot(criterion_list)
plt.plot(index, criterion_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('criterion')

plt.subplot(324)
plt.plot(max_depth_list)
plt.plot(index, max_depth_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('max_depth')

plt.subplot(325)
plt.plot(max_features_list)
plt.plot(index, max_features_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('max_features')

plt.subplot(326)
plt.plot(min_samples_leaf_list)
plt.plot(index, min_samples_leaf_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('min_samples_leaf')
plt.tight_layout()

#visualize optimized tree
regressor = DecisionTreeRegressor(random_state=0,
                                  min_samples_split=hyperopt.space_eval(space,best)['min_samples_split'],
                                  max_features=hyperopt.space_eval(space,best)['max_features'],
                                  max_depth=hyperopt.space_eval(space,best)['max_depth'],
                                  min_samples_leaf=hyperopt.space_eval(space,best)['min_samples_leaf'],
                                   criterion = hyperopt.space_eval(space,best)['criterion'] )
regressor.fit(X1, y)

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydot


dot_data = StringIO()
dot_data= export_graphviz(regressor, out_file='tree.dot')
temp=export_graphviz(regressor,
        out_file=dot_data,
        feature_names=list_features,
        impurity=False,
        filled=True, rounded=True)
graph = pydot.graph_from_dot_data(temp)
graph[0].write_pdf("tree_model_optimized.pdf")


#demonstrate
x_test = np.array([26, 21, 0, 0 ,0 , 0 , 1, 0 , 0])
print(regressor.predict(x_test))

#plot feature importances
fig = plt.figure(figsize=(18,10))
plt.scatter(list_features, regressor.feature_importances_ , s =200*regressor.feature_importances_+75 , c = regressor.feature_importances_  , cmap = 'jet')
plt.title('Feature Importances For Regression Tree')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.tight_layout()









###Tuning Random Forest Regression Using HyperOpt
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import fmin
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


#convert categorical variables to numerical
dataset = pd.read_csv('/media/polatalemdar/9E82D1BB82D197DB/Kaggle/health insurance dataset/insurance.csv')
dataset_modified = pd.get_dummies(dataset, columns=['smoker', 'region', 'sex'], prefix=['smoker', 'region', 'sex'])
dataset_modified= dataset_modified.drop(['smoker_no'] , axis=1)
dataset_modified= dataset_modified.drop(['sex_male'] , axis=1)


n_reps = 5
n_splits = 5
X1= dataset_modified[['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']].values #red
y= dataset_modified['charges'].values
list_features = ['age', 'bmi', 'children', 'smoker_yes', 'region_northeast',
       'region_northwest', 'region_southeast', 'region_southwest',
        'sex_female']

#define objective function to minimize
def objective(space):



    model=RandomForestRegressor(random_state=0,
                          n_estimators=int(space['n_estimators']),
                          max_depth=space['max_depth'] ,
                          min_samples_split=space['min_samples_split'],
                          min_samples_leaf = space['min_samples_leaf'],
                          max_features = space['max_features'],
                          criterion=space['criterion']
                          )
    score_vector = repeated_cv_rmse(X1, y , model, n_reps, n_splits, (space['standardize']==1) )
    score = np.mean(score_vector)
    print("RMSE {} space {}".format(score, space))
    return {'loss': score, 'status': STATUS_OK }

space = {
    'n_estimators': hp.quniform('n_estimators', 3, 53 , 1) ,
    'max_depth': hp.choice('max_depth', [2, 4 , 8 ,16, 32] ),
    'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.9),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.45),
    'max_features': hp.uniform('max_features', 0.1 , 1),
    'criterion': hp.choice('criterion', ['mse', 'mae'] ),
    'standardize':  hp.choice('standardize', [0,1] ),
}


#start searching
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print('Best Model:')
print(hyperopt.space_eval(space,best))

#visualize search
losses_list=trials.losses()
min_samples_split_list=[]
criterion_list = []
max_depth_list = []
max_features_list = []
min_samples_leaf_list = []

for u in range(len(trials.trials)):

    min_samples_split_list.append(trials.trials[u]['misc']['vals']['min_samples_split'])
    criterion_list.append(trials.trials[u]['misc']['vals']['criterion'])
    max_depth_list.append(trials.trials[u]['misc']['vals']['max_depth'])
    max_features_list.append(trials.trials[u]['misc']['vals']['max_features'])
    min_samples_leaf_list.append(trials.trials[u]['misc']['vals']['min_samples_leaf'])

#figure out which iteration the optimal tree was found
print('Optimal Tree Found in iteration number:')
index = losses_list.index(min(losses_list))
print(index)
print('Best Model RMSE:')
print(losses_list[index])

fig= plt.figure(figsize = (15,9))
plt.subplot(321)
plt.plot(losses_list)
plt.plot(index, losses_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('RMSE')

plt.subplot(322)
plt.plot(min_samples_split_list)
plt.plot(index, min_samples_split_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('min_samples_split')

plt.subplot(323)
plt.plot(criterion_list)
plt.plot(index, criterion_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('criterion')

plt.subplot(324)
plt.plot(max_depth_list)
plt.plot(index, max_depth_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('max_depth')

plt.subplot(325)
plt.plot(max_features_list)
plt.plot(index, max_features_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('max_features')

plt.subplot(326)
plt.plot(min_samples_leaf_list)
plt.plot(index, min_samples_leaf_list[index] , 'o', color = 'black')
plt.xlabel('iteration number')
plt.ylabel('min_samples_leaf')
plt.tight_layout()

#visualize optimized tree
regressor = RandomForestRegressor(random_state=0,
                                  n_estimators=int(hyperopt.space_eval(space,best)['n_estimators']),
                                  min_samples_split=hyperopt.space_eval(space,best)['min_samples_split'],
                                  max_features=hyperopt.space_eval(space,best)['max_features'],
                                  max_depth=hyperopt.space_eval(space,best)['max_depth'],
                                  min_samples_leaf=hyperopt.space_eval(space,best)['min_samples_leaf'],
                                   criterion = hyperopt.space_eval(space,best)['criterion'] )
regressor.fit(X1, y)

#plot feature importances
fig = plt.figure(figsize=(18,10))
plt.scatter(list_features, regressor.feature_importances_ , s =200*regressor.feature_importances_+75 , c = regressor.feature_importances_  , cmap = 'jet')
plt.title('Feature Importances For Regression Tree')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.tight_layout()
