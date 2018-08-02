# Import numpy and pandas
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import sklearn.metrics as mc
import os

path="/Users/italopaiva/Documents/Python/Rides/data"


path="//aur.national.com.au/User_Data/AU-VIC-MELBOURNE-3COL1-02/UserData/P695032/Python/Study/Rides"
os.chdir(path)


file_name = "prep_train_data.csv"

#import the file
df = pd.read_csv(file_name, sep='\t', encoding='utf-8')
df = df.drop(df.columns[[0]], axis=1)
#==============================================================================
# df = df[(df['distance'] > 0) & (df['distance'] <= 100) ]
#==============================================================================


#model vars
#==============================================================================
# var = ['distance','pickup_latitude','dropoff_latitude','pickup_longitude',
# 'dropoff_longitude','fare_amount','pickup_range','weekday']
#==============================================================================

#==============================================================================
# var = ['distance','pickup_latitude','dropoff_latitude','pickup_longitude',
#  'dropoff_longitude','weekday']
# 

#==============================================================================
# Create arrays for features and target variable
y = df['fare_amount'].values
#==============================================================================

# X = df[var].values
#==============================================================================
X = df.drop(df.columns[[0]], axis=1)

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
#==============================================================================
# X = X.reshape(-1,1)
#==============================================================================

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

#correlation heat map
#==============================================================================
# sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
#==============================================================================

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# Create the regressor: reg_all
reg_all = LinearRegression() 

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mc.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Plot outputs
plt.scatter(y_pred, y_test,  color='blue')
#==============================================================================
# plt.plot(X_test, y_pred, color='blue', linewidth=3)
#==============================================================================

plt.xticks(())
plt.yticks(())

plt.show()



k1 = pd.DataFrame(y_test)
k2 = pd.DataFrame(y_pred)
k =  pd.concat((k1,k2),axis=1)
k.columns = ['y_test','y_prep']
k['error'] = abs(k['y_test'] - k['y_prep'])
k2 = k.sort_values(by=['error'])




#Cross validation - To reduce risk of overfitting

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg_all,X,y,cv = 3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg_all,X,y,cv = 10)
print(np.mean(cvscores_10))


#Regularization - To reduce risk of overfitting due to parameters with large values

# Instantiate a lasso regressor: lasso
#==============================================================================
# lasso = Lasso(alpha = 0.4,normalize=True)
# 
# # Fit the regressor to the data
# lasso.fit(X,y)
# 
# # Compute and print the coefficients
# lasso_coef = lasso.coef_
# print(lasso_coef)
# 
# # Plot the coefficients
# plt.plot(range(len(df_columns)), lasso_coef)
# plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
# plt.margins(0.02)
#==============================================================================

#Lasso is great for feature selection, 
#but when building regression models, 
#Ridge regression should be your first choice

#function to help with the Ridge normalise analysis

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


#####################

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)