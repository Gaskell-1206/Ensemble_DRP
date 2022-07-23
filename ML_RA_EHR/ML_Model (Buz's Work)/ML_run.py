import pandas as pd
import numpy as np
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")
import os
directory = os.getcwd()
sys.path.append(directory)
from DataModule.Data_Preparation import CoronnaCERTAINDataset
import EvaluationModule as evaluate 
from ModelModule import models

path=directory+'\Dataset'
dataset = CoronnaCERTAINDataset(
    library_root=path,
    challenge="regression_delta_binary", #option: regression, regression_delta, classification, binary_classification
    dataset='CORRONA CERTAIN', 
    process_approach='SC', #option: KVB, SC
    imputation="IterativeImputer", #option: SimpleFill, KNN, SoftImpute, BiScaler, NuclearNormMinimization, IterativeImputer, IterativeSVD, None(raw)
    patient_group='bionaive TNF', #option: "all", "bioexp nTNF", "bionaive TNF", "bionaive orencia", "KVB"
    drug_group='all', #option: "all", "actemra", "cimzia", "enbrel", "humira", "orencia", "remicade", "rituxan", "simponi"
    time_points=(0,3), 
    train_test_rate=0.8,
    remove_low_DAS = True,
    save_csv=True,
    random_state=2022)



train, train_loc = dataset.get_train()
test, test_loc = dataset.get_test()


## train test split 
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]

ensamble_names=[ "Boosting SVM", "Bagging SVM", "Stacking SVM"]
ensamble_parameters= {'ensamble_type':['Boosting','Bagging',"Stacking"],'model_list':[["SVM"]]}
basemodels=[ ["Linear"], ['Ridge'], ["SVM"], ['ANN'],['XGBoost'],['Random Forest'], ['Tree']]
base_names=[ "Linear", 'Ridge', "SVM", 'ANN','XGBoost','Random Forest', 'Tree']
parameters= {'model_list':basemodels}
aml = evaluate.AutoBuild(seed=dataset.random_state, project_name="regression_delta_binary", challenge=dataset.challenge, balance_class=0)

models.test_hyper_parameters(train=train,test=test,dataset=dataset,aml=aml,ensambe_models_params=ensamble_parameters,ensamble_model_ids=ensamble_names,base_model_params=parameters,base_model_ids=base_names, output=directory+'/leaderboard/')

