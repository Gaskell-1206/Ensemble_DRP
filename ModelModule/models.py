from random import random
from re import T
from ModelModule.model_primitive.model_primitive import model_primitive
from ModelModule.BaseModule.base_model import make_model
from ModelModule.BaseModule.base_model import test_hyper_parameters as base_hp_tune
from ModelModule.EnsambleModule.ensamble_model import make_ensamble
from ModelModule.EnsambleModule.ensamble_model import test_hyper_parameters as ensamble_hp_tune
from DataModule.Data_Preparation import CoronnaCERTAINDataset
import os
import sys
import EvaluationModule
directory = os.getcwd()
sys.path.append(directory)
path='../Dataset'
import warnings
warnings.filterwarnings("ignore")
import tqdm
import sklearn.model_selection

def tune_models(dataset_parms, fixed_model_params,test_model_parms, method,project_name,ballance_class, output='../leaderboard/'):
        """"
        dataset_params (dict)- (atribute:value)
        fixed model parms( dict)- (model:(atribute:vales))
        test model parms( dict)- (model:(atribute:vales))
        method (string)- random or grid
        output: for now (list) of tuned models.
        """
        dataset = CoronnaCERTAINDataset(**dataset_parms)
        train=dataset.get_train()[0]
        test= dataset.get_test()[0]
        aml = EvaluationModule.AutoBuild(seed=dataset.random_state, project_name=project_name, challenge=dataset.challenge, balance_class=ballance_class)
        for mod in list(fixed_model_params.keys()):
            temp=model(**fixed_model_params[mod])
            print("base "+temp.model_list[0]+"Done")
            aml.validate(model_id="base "+temp.model_list[0], estimator=temp.model,trainset=train,testset=test)
            temp.hyper_parameter_tune(method=method,dataset=dataset,params=test_model_parms[mod])
            aml.validate(model_id=temp.model_id, estimator=temp.model,trainset=train,testset=test)
        aml.validation_output(dataset=dataset,output=output)
        aml.test_output(dataset=dataset,output=output)
        return aml





def test_models(task_list,project_name, output='../leaderboard/', path=path, ballance_methods=[1]):
    ## task list (dict): ('Challange': {base_model_params=[], base_model_ids=[], ensambe_models_params=[], ensamble_model_ids=[]}) 
    ## ballance methods (list): can contain 0 or 1 or  2
    #for task in tqdm.tqdm(range(len(grid)))
    for task in tqdm.tqdm(range(len(task_list))):
        task=list(task_list.keys())[task]
        dataset = CoronnaCERTAINDataset(
        library_root=path,
        challenge=task, #option: regression, regression_delta, classification, binary_classification
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
        global train
        global test
        train=dataset.get_train()[0]
        test= dataset.get_test()[0]
        #print("passed 1")
        for i in tqdm.tqdm(range(len(ballance_methods))):
            #print("passed 2")
            aml = EvaluationModule.AutoBuild(seed=dataset.random_state, project_name=project_name, challenge=dataset.challenge, balance_class=ballance_methods[i])
            test_models_single_task( train=train, test=test, dataset=dataset, aml=aml, output=output, **task_list[task])


def test_models_single_task( train, test, dataset, aml, base_model_params=[], base_model_ids=[], ensambe_models_params=[], ensamble_model_ids=[], output='../leaderboard/'):
    if(base_model_params!=[]):
        base_hp_tune(parameters=base_model_params,train=train,test=test,dataset=dataset,aml=aml,name_list=base_model_ids, output=output)
    if(ensambe_models_params!=[]):
        ensamble_hp_tune(parameters=ensambe_models_params,train=train,test=test,dataset=dataset,aml=aml,name_list=ensamble_model_ids,output=output)
    aml.validation_output(dataset=dataset,output=output)
    aml.test_output(dataset=dataset,output=output)


class model():
    def __init__(self, model_list,base_model_params=[], ensamble_type='Stacking',model_id="", meta_learner=None, meta_learner_params=None, hyper_parameters=None, challenge="Regression", ensamble=False, random_state=0):
        self.random_state=random_state
        self.challenge=challenge
        self.model_id=model_id
        self.model_list=model_list
        if(ensamble):
            self.model=make_ensamble(model_list, base_model_params, ensamble_type, model_id, meta_learner, meta_learner_params, hyper_parameters, challenge).model
        else:
            self.model=make_model(model_id,model_list, base_model_params, challenge).model
    def random_search_model(self,params,dataset):
        if(self.challenge=="Reggression"):
           scoring = "f1_weighted"
        else:
            scoring= "neg_mean_squared_error"
            ## look into the cv. try stratfied k fold. 
        model=sklearn.model_selection.RandomizedSearchCV(estimator=self.model,param_distributions=params,random_state=self.random_state, scoring=scoring)
        train=dataset.get_train()[0]
        test= dataset.get_test()[0]
        X_train = train.iloc[:,:-1]
        y_train = train.iloc[:,-1]
        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:,-1]
        self.model=model.fit(X_train,y_train).best_estimator_

    def grid_search_model(self,params,dataset):
        if(self.challenge=="Reggression"):
           scoring = "f1_weighted"
        else:
            scoring= "neg_mean_squared_error"
        
        model=sklearn.model_selection.GridSearchCV(estimator=self.model,param_grid=params, scoring=scoring)
        train=dataset.get_train()[0]
        test= dataset.get_test()[0]
        X_train = train.iloc[:,:-1]
        y_train = train.iloc[:,-1]
        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:,-1]
        self.model=model.fit(X_train,y_train).best_estimator_
    def hyper_parameter_tune(self,params,method,dataset):
        train=dataset.get_train()[0]
        test= dataset.get_test()[0]
        X_train = train.iloc[:,:-1]
        y_train = train.iloc[:,-1]
        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:,-1]
        if(method=="random"):
            self.random_search_model(params=params,dataset=dataset)
        else:
            self.grid_search_model(params=params,dataset=dataset)
