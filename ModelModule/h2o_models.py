import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

def h2o_model(dataset):
    # read train, test from dataloader
    train_set, train_loc = dataset.get_train()
    test_set, test_loc = dataset.get_test()
    # Start the H2O cluster (locally)
    h2o.init()

    # Import a sample binary outcome train/test set into H2O
    # train_h2o = h2o.upload_file(str(train_loc))
    # test_h2o = h2o.upload_file(str(test_loc))
    train_h2o = h2o.import_file(str(train_loc))
    test_h2o = h2o.import_file(str(test_loc))

    # Identify predictors and response
    x = train_h2o.columns[:-1]
    # y = "DAS28_CRP_3M"
    y = dataset.target

    for feature in dataset.categorical:
        train_h2o[feature] = train_h2o[feature].asfactor()
        test_h2o[feature] = test_h2o[feature].asfactor()

    if "classification" in dataset.challenge:
        train_h2o[y] = train_h2o[y].asfactor()
        test_h2o[y] = test_h2o[y].asfactor()

    # Train a GBM model setting nfolds to 10
    
    # Define GBM model
    gbm = H2OGradientBoostingEstimator(nfolds = 10, seed = dataset.random_state)
    # Define GBM hyperparameters
    gbm_params = {'learn_rate': [i * 0.01 for i in range(1, 11)],
                  'max_depth': list(range(2, 11)),
                  'sample_rate': [i * 0.1 for i in range(5, 11)],
                  'col_sample_rate': [i * 0.1 for i in range(1, 11)]}
    search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 50, 'seed': dataset.random_state}
    gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid',
                          hyper_params=gbm_params,
                          search_criteria=search_criteria)
    gbm_grid.train(x=x, y=y, training_frame=train_h2o)
    gbm_gridperf = gbm_grid.get_grid(sort_by='mse', decreasing=False)
    best_gbm = gbm_gridperf.models[0]
    print(best_gbm.mse())
    