import h2o
h2o.init()
import os
from h2o.automl import H2OAutoML
from h2o.estimators.infogram import H2OInfogram
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr
import sys
import csv
from pathlib import Path
sys.path.insert(0, '/gpfs/home/sc9295/Projects/Ensemble_DRP')
from DataModule.Data_Preparation import CoronnaCERTAINDataset
import ModelModule.EvaluationModule as EvaluationModule
pd.options.mode.chained_assignment = None

def Pearson_Correlation(true, pred):
    return pearsonr(true, pred)[0][0]

def F1_Score(true, pred):
    return metrics.f1_score(true, pred, average='weighted')

def Classification_Accuracy(true, pred):
    return metrics.accuracy_score(true, pred)

def add_model_family(row):
    if "StackedEnsemble" in row['model_id']:
        return '_'.join(row['model_id'].split('_',3)[:2])
    else:
        return row['model_id'].split('_')[0]

def add_ensemble(row):
    if "StackedEnsemble" in row['model_id']:
        return "ALL"
    else:
        return "N/A"

def add_meta_learner(row):
    if "StackedEnsemble" in row['model_id']:
        return "glm"
    else:
        return "N/A"

def get_h2o_model(model_family):
    model_id = base_models['model_id'][model_family]
    # model_path_dir = './leaderboard_new/leaderboard/model_saved/'
    # model_path = os.path.join(model_path_dir,model_id)
    # uploaded_model = h2o.upload_model(model_path)
    model = h2o.get_model(model_id)
    return model

# get evaluation metrics list
def get_metrics_list(model_id_list):
    header = ['challenge', 'imputation', 'model_id','model_family', 'ensemble', 'metalearner']
    test_header = ['test_acc', 'test_F1', 'test_auc', 'test_aucpr']
    model = h2o.get_model(model_id_list[0])
    time_stamp = '_'.join(model_id_list[0].split('_')[-2:])
    cv_df = model.cross_validation_metrics_summary().as_data_frame().set_index('')
    evaluation_metrics_list = list(cv_df.index)
    # evaluation_metrics_list = ['accuracy','auc','f0point5','f1','f2','max_per_class_error','pr_auc','']
    evaluation_metrics_mean = [i+'_mean' for i in evaluation_metrics_list]
    evaluation_metrics_std = [i+'_std' for i in evaluation_metrics_list]
    header_list = header + test_header + evaluation_metrics_mean + evaluation_metrics_std
    return header_list, evaluation_metrics_list, time_stamp

def get_test_performance(model, test, test_h2o, results_df, i):
    true = test[dataset.target]
    pred = model.predict(test_h2o).as_data_frame()['predict']
    baseline = test['DAS28_CRP_0M']
    results_df.loc[i,'test_acc'] = Classification_Accuracy(true, pred)
    results_df.loc[i,'test_F1'] = F1_Score(true, pred)
    results_df.loc[i,'test_auc'] = model.model_performance(test_h2o).auc()
    results_df.loc[i,'test_aucpr'] = model.model_performance(test_h2o).aucpr()

    baseline, true, pred = np.array(baseline), np.array(true), np.squeeze(np.array(pred))
    performance = pd.DataFrame(list(zip(baseline, true, pred)),
                      columns=['baseline', 'true', 'pred'])

    return results_df

# define data module
def get_data():
    dataset = CoronnaCERTAINDataset(
        library_root = '/gpfs/home/sc9295/Projects/Ensemble_DRP/Dataset',
        challenge = "binary_classification", #option: regression_delta, 3_classification, binary_classification
        dataset = 'CORRONA CERTAIN', 
        imputation = "KNN", #option: SimpleFill, KNN, IterativeImputer, None(raw)
        patient_group = "bionaive TNF", #option: "all", "bioexp nTNF", "bionaive TNF", "bionaive orencia"
        drug_group = 'all', #option: "all", "actemra", "cimzia", "enbrel", "humira", "orencia", "remicade", "rituxan", "simponi"
        time_points = (0,3), 
        train_test_rate = 0.7,
        save_csv = False, 
        random_state = 42,
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

    return dataset, train_h2o, test_h2o, train, test, x, y

if __name__ == "__main__":
    dataset, train_h2o, test_h2o, train, test, x, y = get_data()

    # AutoML configuration
    project_name = f"binary_classification_Sep7_{dataset.imputation}"
    csv_file_name = "binary_classification_Sep7"
    nfolds = 10
    sample_factors = [1.0,0.5]

    if "regression" in dataset.challenge:
        # aml = H2OAutoML(max_models=10, max_runtime_secs=60, nfolds=nfolds, seed = dataset.random_state, project_name = project_name, keep_cross_validation_predictions=True)
        aml = H2OAutoML(nfolds=nfolds, seed = dataset.random_state, project_name = project_name, keep_cross_validation_predictions=True)
    elif dataset.challenge == "binary_classification":
        aml = H2OAutoML(nfolds=nfolds, balance_classes=True, class_sampling_factors=sample_factors, sort_metric='mean_per_class_error', seed = dataset.random_state, project_name = project_name, keep_cross_validation_predictions=True)
    elif dataset.challenge == "classification":
        aml = H2OAutoML(nfolds=nfolds, balance_classes=True, sort_metric='mean_per_class_error', seed = dataset.random_state, project_name = project_name, keep_cross_validation_predictions=True)

    # train models
    aml.train(x=x, y=y, training_frame=train_h2o)
    
    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print("---------------AutoML Leaderboard:", lb.head(rows=lb.nrows))
    print("---------------Best Model:", aml.leader)

    model_id_list = lb.as_data_frame()['model_id'].values.tolist()

    # base_model_dict = {}
    base_model_list = []
    base_model_family_list = []
    base_model_id_list = []
    base_model_family_id_list = []
    # get header and evaluation metrics list
    header_list, evaluation_metrics_list, time_stamp = get_metrics_list(model_id_list)

    # cross-validation resutls
    # global results_df, i
    results_df = pd.DataFrame(columns=header_list)
    
    for model_id in model_id_list:
        i = len(results_df)
        results_df.loc[i,'model_id'] = model_id

        model = h2o.get_model(model_id)
        my_local_model = h2o.download_model(model, path="/gpfs/home/sc9295/Projects/Ensemble_DRP/leaderboard_new/model_saved")
        
        results_df = get_test_performance(model, test, test_h2o, results_df, i)
        
        cv_df = model.cross_validation_metrics_summary().as_data_frame().set_index('')

        for metrics_ in evaluation_metrics_list:
            try:
                metrics_mean = cv_df.loc[metrics_,'mean']
                metrics_std = np.std(cv_df.loc[metrics_,'cv_1_valid':f'cv_{nfolds}_valid'])
                results_df.loc[i,metrics_+'_mean'] = metrics_mean
                results_df.loc[i,metrics_+'_std'] = metrics_std
            except KeyError:
                pass
        # i+=1
    
    results_df.loc[:,'model_family'] = results_df.apply(lambda row:add_model_family(row),axis=1)
    results_df.loc[:,'ensemble'] = results_df.apply(lambda row:add_ensemble(row),axis=1)
    results_df.loc[:,'metalearner'] = results_df.apply(lambda row:add_meta_learner(row),axis=1)

    # select the best model of each family
    selected_models = results_df.sort_values(by=f'auc_mean',ascending=True).groupby('model_family').first()
    stacked_models = selected_models.loc[selected_models.index.str.startswith(('StackedEnsemble')),:]
    base_models = selected_models.loc[~selected_models.index.str.startswith(('StackedEnsemble')),:]
    # select top 3 base models
    selected_base_models = base_models.loc[(base_models['auc_mean'] < base_models['auc_mean'].mean()),:]
    for base_model in selected_base_models.index:
        base_model_list.append(get_h2o_model(base_model))
        base_model_id_list.append(base_model)
    for base_model in base_models.index:
        base_model_family_list.append(get_h2o_model(base_model))
        base_model_family_id_list.append(base_model)

    # customize ensemble models
    metalearner_list = ['glm','gbm','drf','deeplearning','xgboost']

    for metalearner in metalearner_list:
        i = len(results_df)
        ensemble = H2OStackedEnsembleEstimator(metalearner_algorithm=metalearner,
                                               metalearner_nfolds=nfolds,
                                               base_models=base_model_list)
        ensemble.train(x=x, y=y, training_frame=train_h2o)
        cv_df = ensemble.cross_validation_metrics_summary().as_data_frame().set_index('')

        results_df = get_test_performance(ensemble, test, test_h2o, results_df, i)

        results_df.loc[i,'model_id'] = f'StackedEnsemble_SelectedModels_1_SC_1_{time_stamp}'
        results_df.loc[i,'ensemble'] = '+'.join(base_model_id_list)
        results_df.loc[i,'metalearner'] = metalearner

        for metrics_ in evaluation_metrics_list:
            try:
                metrics_mean = cv_df.loc[metrics_,'mean']
                metrics_std = np.std(cv_df.loc[metrics_,'cv_1_valid':f'cv_{nfolds}_valid'])
                results_df.loc[i,metrics_+'_mean'] = metrics_mean
                results_df.loc[i,metrics_+'_std'] = metrics_std
            except KeyError:
                pass

        # add BestOfFamilies
        i = len(results_df)
        ensemble = H2OStackedEnsembleEstimator(metalearner_algorithm=metalearner,
                                               metalearner_nfolds=nfolds,
                                               base_models=base_model_family_list)
        ensemble.train(x=x, y=y, training_frame=train_h2o)
        cv_df = ensemble.cross_validation_metrics_summary().as_data_frame().set_index('')

        results_df = get_test_performance(ensemble, test, test_h2o, results_df, i)

        results_df.loc[i,'model_id'] = f'StackedEnsemble_SelectedModels_1_SC_1_{time_stamp}'
        results_df.loc[i,'ensemble'] = '+'.join(base_model_family_id_list)
        results_df.loc[i,'metalearner'] = metalearner

        for metrics_ in evaluation_metrics_list:
            try:
                metrics_mean = cv_df.loc[metrics_,'mean']
                metrics_std = np.std(cv_df.loc[metrics_,'cv_1_valid':f'cv_{nfolds}_valid'])
                results_df.loc[i,metrics_+'_mean'] = metrics_mean
                results_df.loc[i,metrics_+'_std'] = metrics_std
            except KeyError:
                pass

    # add labels
    results_df.loc[:,'model_family'] = results_df.apply(lambda row:add_model_family(row),axis=1)
    results_df.loc[:,'challenge'] = dataset.challenge
    results_df.loc[:,'imputation'] = dataset.imputation
    results_df = results_df.reset_index(drop=True)

    # final output
    output = Path('/gpfs/home/sc9295/Projects/Ensemble_DRP/leaderboard_new')
    os.makedirs(output, exist_ok=True)
    save_path = output / f'{csv_file_name}_output.csv'

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