import h2o
import os
from h2o.automl import H2OAutoML
from h2o.estimators.infogram import H2OInfogram
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import pandas as pd
import numpy as np
import sys
import csv
from pathlib import Path
sys.path.insert(0, '/gpfs/home/sc9295/Projects/ML_RA_EHR')
from DataModule.Data_Preparation import CoronnaCERTAINDataset
import EvaluationModule
import EvaluationModule_H2O
pd.options.mode.chained_assignment = None

if __name__ == "__main__":
    # define data module
    dataset = CoronnaCERTAINDataset(
        library_root = '/gpfs/home/sc9295/Projects/ML_RA_EHR/Dataset/',
        challenge = 'classification', #option: regression, regression_delta, classification, binary_classification, regression_delta_binary
        dataset = 'CORRONA CERTAIN', 
        process_approach = 'SC', #option: KVB, SC
        imputation = None, #option: SimpleFill, KNN, SoftImpute, BiScaler, NuclearNormMinimization, IterativeImputer, IterativeSVD, None(raw)
        patient_group = ['bionaive TNF'], #option: "all", "bioexp nTNF", "bionaive TNF", "bionaive orencia", "KVB"
        drug_group = 'all', #option: "all", "actemra", "cimzia", "enbrel", "humira", "orencia", "remicade", "rituxan", "simponi"
        time_points = (0,3), 
        train_test_rate = 0.8,
        remove_low_DAS = True,
        save_csv = False, 
        random_state = 2022,
        verbose=False)

    train, train_loc = dataset.get_train()
    test, test_loc = dataset.get_test()
    # Start the H2O cluster (locally)
    h2o.init()

    # Import a sample binary outcome train/test set into H2O
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

    # Run AutoML for 20 base models
    project_name = "SC_3_Class_classification_Aug4_final"
    nfolds = 10
    sample_factors = [1.0,0.5]
    if "regression" in dataset.challenge:
        aml = H2OAutoML(nfolds=nfolds, seed = 2022, project_name = project_name)
    elif dataset.challenge == "binary_classification":
        aml = H2OAutoML(nfolds=nfolds, balance_classes=True, class_sampling_factors=sample_factors, sort_metric='mean_per_class_error', seed = 2022, project_name = project_name)
    elif dataset.challenge == "classification":
        aml = H2OAutoML(nfolds=nfolds, balance_classes=True, sort_metric='mean_per_class_error', seed = 2022, project_name = project_name)
    # max_runtime_secs
    aml.train(x=x, y=y, training_frame=train_h2o)
    
    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)
    print(aml.leader)

    model_id_list = lb.as_data_frame()['model_id'].values.tolist()

    # model_id_prefix = []
    # model_id_list = []
    # for model_id in lb.as_data_frame()['model_id'].values.tolist():
    #     if model_id.split('_')[0] not in model_id_prefix:
    #         model_id_prefix.append(model_id.split('_')[0])
    #         model_id_list.append(model_id)
    #     if len(model_id_list) > 10:
    #         break

    header = ['dataset','challenge', 'model_id']
        
    # get evaluation metrics list
    model = h2o.get_model(model_id_list[0])
    cv_df = model.cross_validation_metrics_summary().as_data_frame().set_index('')
    evaluation_metrics_list = list(cv_df.index)
    evaluation_metrics_mean = [i+'_mean' for i in evaluation_metrics_list]
    evaluation_metrics_std = [i+'_std' for i in evaluation_metrics_list]
    header_list = header + evaluation_metrics_mean + evaluation_metrics_std

    # cross-validation resutls
    results_df = pd.DataFrame(columns=header_list)
    i=0
    for model_id in model_id_list:
        results_df.loc[i,'model_id'] = model_id
        results_df.loc[i,'dataset'] = 'val'
        model = h2o.get_model(model_id)
        my_local_model = h2o.download_model(model, path="/gpfs/home/sc9295/Projects/ML_RA_EHR/leaderboard/model_saved")
        cv_df = model.cross_validation_metrics_summary().as_data_frame().set_index('')

        for metrics in evaluation_metrics_list:
            try:
                metrics_mean, metrics_std = cv_df.loc[metrics,'mean'], np.std(cv_df.loc[metrics,'cv_1_valid':f'cv_{nfolds}_valid'])
                results_df.loc[i,metrics+'_mean'] = metrics_mean
                results_df.loc[i,metrics+'_std'] = metrics_std
            except KeyError:
                pass
        i+=1
        
        results_df.loc[:,'challenge'] = dataset.challenge
    
    # test results
    
    output = Path('/gpfs/home/sc9295/Projects/ML_RA_EHR/leaderboard')
    os.makedirs(output, exist_ok=True)
    save_path = output / f'{project_name}_output.csv'

    # if header exists
    has_header = False
    try:
        if len(pd.read_csv(save_path)) > 0:
            has_header = True
    except:
        has_header = False
    
    # save validation output to csv
    with open(save_path, 'a+') as f:
        # create the csv writer
        writer = csv.writer(f)
        if not has_header:
            # write the header
            writer.writerow(header_list)
        for index, rows in results_df.iterrows():
            # write the data
            writer.writerow(rows)
        print(f"save output in csv at {save_path}")