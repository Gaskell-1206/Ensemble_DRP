import sys
sys.path.insert(0, '..')
import DataModule.Data_Preparation
import EvaluationModule 
import EnsambleModule.ensamble_model
import os
import sklearn
from itertools import chain, combinations

dataset = DataModule.Data_Preparation.CoronnaCERTAINDataset(
    library_root="../Dataset",
    challenge="binary_classification", #option: regression, regression_delta, classification, binary_classification
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
len(y_test)+len(y_train)



train, train_loc = dataset.get_train()
test, test_loc = dataset.get_test()


## train test split 
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,:-1]

y_test = test.iloc[:,-1]
meta_learners=[ "Bayes", 'Logistic', "SVM", 'ANN','XGBoost','Random Forest', 'Tree']
from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r+1) for r in range(len(s)))
estemators=[list(c) for c in list(powerset(meta_learners))]

aml=EvaluationModule.AutoBuild(seed=1,challenge='binary_classification')
parameters= {'ensamble_type':["Stacking"],'model_list':estemators[1:4], 'challange':['Classification'], 'meta_learner':meta_learners[1:4]}
EnsambleModule.ensamble_model.test_hyper_parameters(parameters,train,test,dataset,aml)