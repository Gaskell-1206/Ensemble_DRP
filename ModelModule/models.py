from random import random
from re import T

import h2o
from ModelModule.model_primitive.model_primitive import model_primitive
from ModelModule.BaseModule.base_model import make_model
from ModelModule.BaseModule.base_model import test_hyper_parameters as base_hp_tune
from ModelModule.EnsambleModule.ensamble_model import make_ensamble
from ModelModule.EnsambleModule.ensamble_model import test_hyper_parameters as ensamble_hp_tune
from DataModule.Data_Preparation import CoronnaCERTAINDataset
import os
import sys
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import EvaluationModule
from sklearn.model_selection import (KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, ShuffleSplit,
                                     StratifiedKFold)

directory = os.getcwd()
sys.path.append(directory)
path='../Dataset'
import warnings
warnings.filterwarnings("ignore")
import tqdm
import sklearn.model_selection
import pickle
from h2o.grid.grid_search import H2OGridSearch



def tune_models(dataset_parms, fixed_model_params,test_model_parms, method,project_name,ballance_class,search_criteria, output='../leaderboard/'):
        """"
        dataset_params (dict)- (atribute:value)
        fixed model parms( dict)- (model:(atribute:vales))
        test model parms( dict)- (model:(atribute:vales))
        method (string)- random or grid
        output: for now (list) of tuned models.
        """
        output=[]
        for ballance in ballance_class:
            for task in dataset_parms:
                print("ballance is", ballance)
                print("task is:", task)
                output.append(tune_model_singluar(dataset_parms=task, fixed_model_params=fixed_model_params,test_model_parms=test_model_parms, method=method,project_name=project_name,ballance_class=ballance,search_criteria=search_criteria, output='../leaderboard/')       
                )

def tune_model_singluar(dataset_parms, fixed_model_params,test_model_parms, method,project_name,ballance_class, search_criteria,output='../leaderboard/'):
        """"
        dataset_params (dict)- (atribute:value)
        fixed model parms( dict)- (model:(atribute:vales))
        test model parms( dict)- (model:(atribute:vales))
        method (string)- random or grid
        output: for now (list) of tuned models.
        """
        print("data set params are ", dataset_parms)
        dataset = CoronnaCERTAINDataset(**dataset_parms)
        train=dataset.get_train()[0]
        test= dataset.get_test()[0]
        aml = EvaluationModule.AutoBuild(seed=dataset.random_state, project_name=project_name, challenge=dataset.challenge, balance_class=ballance_class)
        for mod in list(fixed_model_params.keys()):
            if(fixed_model_params[mod]['ensamble'] and fixed_model_params[mod]['h20']):
                temp=model(**fixed_model_params[mod])
                #temp.hyper_parameter_tune(method=method,dataset=dataset,params=test_model_parms[mod],search_criteria=search_criteria)
                aml.validate(model_id=temp.model_id, estimator=temp.model,trainset=train,testset=test, h20=temp.h20,model_params=fixed_model_params)
                #temp.save_model(adtioanl_name="ballance class="+str(aml.balance_class)+", challange="+dataset.challenge)
            else:
                temp=model(**fixed_model_params[mod])
                #temp.hyper_parameter_tune(method=method,dataset=dataset,params=test_model_parms[mod], search_criteria=search_criteria)
                aml.validate(model_id=temp.model_id, estimator=temp.model,trainset=train,testset=test, h20=temp.h20,model_params=fixed_model_params)
                #temp.save_model(adtioanl_name="ballance class="+str(aml.balance_class)+", challange="+dataset.challenge)
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
    return aml.validation_output(dataset=dataset,output=output)
    #aml.test_output(dataset=dataset,output=output)


class model():
    def __init__(self, model_list,ensamble,base_model_params=[],
     ensamble_type='Stacking',model_id="", meta_learner=None, 
     meta_learner_params=None, hyper_parameters=None, challenge="Regression", 
      random_state=0, h20=True, dataset_parms="", frame=None
    ):
        self.base_model_params=base_model_params
        self.random_state=random_state
        self.challenge=challenge
        self.model_id=model_id
        self.model_list=model_list
        self.h20=h20
        self.ensamble=ensamble
        if(ensamble):
            #if((dataset_parms==None and frame==None) and h20):
            #    raise ValueError('h2o stacked ensmable requires a dataset to train base models')
            self.model=make_ensamble(model_list, base_model_params, ensamble_type, model_id, meta_learner, meta_learner_params, hyper_parameters, challenge, h20,dataset_parms,frame).model
        else:
            self.model=make_model(model_id,model_list, base_model_params, challenge, h20).model
        #for model in model_list:
        #    make_model(model, base_model_params, challenge)

    
    def random_search_model(self,dataset,params=[]):
        train, train_loc = dataset.get_train()
        if(self.h20):
            train_h2o = h2o.upload_file(str(train_loc))
            x = train_h2o.columns[:-1]
            y=dataset.target
            for feature in dataset.categorical:
                train_h2o[feature] = train_h2o[feature].asfactor()
                if(self.challenge!="Regression"):
                    train_h2o[y] = train_h2o[y].asfactor()
            self.model=self.model.train(x=x, y=y, training_frame=train_h2o)
        else:
            if(self.challenge=="Regression"):
                scoring = "f1_weighted"
            else:
                scoring= "neg_mean_squared_error"
                ## look into the cv. try stratfied k fold. 
            model=sklearn.model_selection.RandomizedSearchCV(estimator=self.model,param_distributions=params, scoring=scoring,cv=3)
            X_train = train.iloc[:,:-1]
            y_train = train.iloc[:,-1]
            X_test = test.iloc[:,:-1]
            y_test = test.iloc[:,-1]
            self.model=model.fit(X_train,y_train).best_estimator_
            print("best_params:",model.best_params_)

    def grid_search_model(self,params,dataset):
        if(self.challenge=="Regression"):
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
        print("best_params:",model.best_params_)


    def hyper_parameter_tune(self,params,method,dataset, search_criteria):
        if(self.h20==False):
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
        else:
            train=dataset.get_train()[0]
            train.to_csv('../Dataset/temp.csv')
            train_h2o = h2o.upload_file('../Dataset/temp.csv')
            train_h2o.drop(['C1'], axis=1)
            x_h20 = train_h2o.columns[:-1]
            y_h20 = dataset.target
            for feature in ['grp','init_group','gender','final_education','race_grp','ethnicity','newsmoker','drinker','ara_func_class']:
                train_h2o[feature] = train_h2o[feature].asfactor()
            if(self.challenge!="Regression"):
                train_h2o[y_h20] = train_h2o[y_h20].asfactor()
            if(self.ensamble==False):
                self.model = H2OGridSearch(model=self.model,
                        hyper_params=params,
                        search_criteria=search_criteria)
                temp=self.model.train(x=x_h20, y=y_h20, training_frame=train_h2o) 
                self.model=temp.get_grid(sort_by='mse', decreasing=True).models[0]
            else:
                ###take a break and coime back the goal is to figure out how to tune the hyper-params like that .
                here=[]
                i=0
                for mod in params:
                    temp=model(model_list=[mod], challenge=self.challenge, base_model_params=[self.base_model_params[i]], ensamble=False)
                    i=i+1
                    if(bool(params[mod])):
                        temp.hyper_parameter_tune(params=params[mod], method="random", dataset=dataset,search_criteria=search_criteria)
                        if(hasattr(temp.model, 'model_ids')==False):
                            here.append(temp.model)
                        else:    
                            here=here+temp.model.model_ids
                    else:
                        temp.random_search_model(dataset=dataset)
                        if(hasattr(temp.model, 'model_ids')==False):
                            here.append(temp.model)
                        else:
                            here.append(temp.model.model_id)
                self.model=H2OStackedEnsembleEstimator(base_models=here)
                self.model.train(x=x_h20, y=y_h20, training_frame=train_h2o)






    def save_model(self,adtioanl_name="",replace_name=False, path=r'../Saved_Models/'):
        if(adtioanl_name!=""):
            if(replace_name):
                filename=path+adtioanl_name+'.sav'
            else:
                 filename=path+adtioanl_name+" "+ self.model_id+'.sav'
        else:
            filename=path+self.model_id+'.sav'
        if(self.h20):
            print("path is," , path[:-1])
            h2o.download_model(model=self.model, path=path[:-1])
        else:    
            pickle.dump(self.model, open(filename, 'wb'))
