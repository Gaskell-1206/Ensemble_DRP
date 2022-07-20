import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def fine_tune(train, model, search_methods):
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]

    # use paramaters combination of each model
    if model == "drf_classification": # distributed random forests
        model = RandomForestClassifier()
        param_search_grid = rf_parameters()
        scoring = "f1_weighted"
    elif model == "drf_regression":
        model = RandomForestRegressor()
        param_search_grid = rf_parameters()
        scoring = "neg_mean_squared_error"
    elif model == "GBM":
        pass # to be added
    
    print("param_search_grid:", param_search_grid)
    
    if search_methods == "GridSearch":
        search_model = GridSearchCV(estimator=model, param_grid=param_search_grid, scoring=scoring,
                                    cv=10, n_jobs=-1, verbose = 0)
    elif search_methods == "RandomSearch":
        search_model = RandomizedSearchCV(estimator=model, param_distributions=param_search_grid, scoring=scoring,
                                          n_iter=30, cv=10, n_jobs=-1, verbose = 0)
    search_model.fit(X_train, y_train)
    print("best_params:",search_model.best_params_)
    tuned_model = search_model.best_estimator_
    return tuned_model

def rf_parameters():
    # define paramter for RF
    # Number of samples to draw from X to train each base estimator
    max_samples = [float(x) for x in np.linspace(start = 0.5, stop = 1, num =6)]
    max_samples.append(None)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_search_grid = {
        'max_samples': max_samples,
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}
    return param_search_grid