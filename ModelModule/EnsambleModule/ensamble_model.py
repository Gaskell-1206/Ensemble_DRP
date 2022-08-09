
from pandas.api.types import is_dict_like
import sys
import tqdm
import warnings
import sklearn as sk
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree 
from sklearn import model_selection
import xgboost as xgb
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from DataModule.Data_Preparation import CoronnaCERTAINDataset
import h2o

warnings.filterwarnings("ignore")
sys.path.insert(0, '..')
import EvaluationModule 
from ModelModule.model_primitive.model_primitive import model_primitive


def test_hyper_parameters(parameters, train,test,dataset, aml, name_list=[], output='../leaderboard/'):
    grid=list(model_selection.ParameterGrid(parameters))
    for params in (range(len(grid))):
        if(name_list==[]):
            model_id=str(grid[params])
        else: 
            model_id=(name_list[params])
        model=make_ensamble(**grid[params]).model
        aml.validate(model_id,model,train,test)
        aml.evaluate(model_id=model_id,model=model,dataset="test",working_set=test, save_df= aml.train_perf  )
    # aml.validation_output(dataset=dataset, output=output)
    # aml.test_output(dataset=dataset, output=output)

class make_ensamble(model_primitive): 
    def __init__(self, model_list,base_model_params=[], ensamble_type='Stacking',model_id="", meta_learner=None, meta_learner_params=None, hyper_parameters=None, challenge="Regression", h20=True, dataset_parms=None, frame=None):
        ## exception cehcking. 
        model_primitive.__init__(self,model_id,model_list,base_model_params,challenge, h20)
        temp=dict()
        for i in range(len(model_list)):
            if(i>=len(base_model_params)):
                base_model_params.append({})
            temp[model_list[i]]=base_model_params[i]
        model_list=temp
        ## intitlizing atrivutes    
        self.base_model_list=model_list
        if is_dict_like(hyper_parameters)==False and hyper_parameters!=None:
            raise ValueError("hyper_parameters must be dict like")
        if meta_learner not in ( 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes",'ANN','XGBoost','Random Forest','Tree', None ):
            raise ValueError(meta_learner, "not right meta_learner must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso','ANN', "SVM", "Bayes",'XGBoost','Random Forest','Tree')
        if ensamble_type not in ('Stacking', 'Bagging', 'Boosting'):
            raise ValueError("ensamble_type must be in ", 'Stacking', 'Bagging', 'Boosting')
        if len(model_list)!=1 and ensamble_type in ( 'Bagging', 'Boosting'):
            raise ValueError(ensamble_type, 'requires a homogeneous weak learner ')
        self.n_weak_learners=len(model_list)
        self.ensamble_type=ensamble_type
        self.meta_learner=meta_learner
        
        if hyper_parameters==None:
            hyper_parameters=dict()
        self.hyper_parameters=hyper_parameters
        

        ## dictionary for base models split between classfication and regression. 

        # list of tuples for stacking  
        estemator=[]
        for model in self.base_model_list:
            temp=self.model_dict[model]
            temp.set_params(**self.base_model_list[model])
            estemator.append((model,temp))
        self.estemators=estemator 
        ## difinging model type
        if self.challenge=='Regression':
            if self.ensamble_type=="Bagging":
                self.model=ensemble.BaggingRegressor(base_estimator=self.estemators[0][1],**self.hyper_parameters)
            elif self.ensamble_type=="Boosting":
                self.model=ensemble.AdaBoostRegressor(base_estimator=self.estemators[0][1],**self.hyper_parameters)
            elif(self.ensamble_type=="Stacking" and h20==True):
                if(dataset_parms!=None):
                    print("ensamble dataset params are ",dataset_parms )
                    dataset = CoronnaCERTAINDataset(**dataset_parms[0])
                    train, train_loc = dataset.get_train()
                    train_h2o = h2o.upload_file(str(train_loc))
                    x_h2o = train_h2o.columns[:-1]
                    y_h20=dataset.target
                    for feature in dataset.categorical:
                        train_h2o[feature] = train_h2o[feature].asfactor()
                else:
                    train_h2o=frame
                    x_h2o = train_h2o.columns[:-1]
                    y_h20=train_h2o.columns[-1]
                bm=[]
                for model in range(len(self.estemators)):
                    bm.append(self.estemators[model][1].train(x=x_h2o, y=y_h20, training_frame=train_h2o))

                self.model=H2OStackedEnsembleEstimator(base_models=bm, **self.hyper_parameters)
            else:
                if self.meta_learner!=None:
                    temp=self.model_dict[self.meta_learner]
                    if(meta_learner_params!=None):
                        temp.set_params(**meta_learner_params)
                    self.model=ensemble.StackingRegressor(estimators=self.estemators,final_estimator=temp,**self.hyper_parameters)
                else:
                    self.model=ensemble.StackingRegressor(estimators=self.estemators,**self.hyper_parameters)
        else: 

            if self.ensamble_type=="Bagging":
                self.model=ensemble.BaggingClassifier(base_estimator=self.estemators[0][1],**self.hyper_parameters)
            elif self.ensamble_type=="Boosting":
                self.model=ensemble.AdaBoostClassifier(base_estimator=self.estemators[0][1],**self.hyper_parameters)

            elif(self.ensamble_type=="Stacking" and h20):
                if(dataset_parms!=None):
                    dataset = CoronnaCERTAINDataset(**dataset_parms)
                    train, train_loc = dataset.get_train()
                    train_h2o = h2o.upload_file(str(train_loc))
                    x_h2o = train_h2o.columns[:-1]
                    y_h20=dataset.target
                    for feature in dataset.categorical:
                       train_h2o[feature] = train_h2o[feature].asfactor()
                    train_h2o[y_h20] = train_h2o[y_h20].asfactor()
                else:
                    train_h2o=frame
                    x_h2o = train_h2o.columns[:-1]
                    y_h20=train_h2o.columns[-1]
                bm=[]
                for model in range(len(self.estemators)):
                    self.estemators[model][1].train(x=x_h2o, y=y_h20, training_frame=train_h2o)
                    bm.append(self.estemators[model][1])
                    self.model=H2OStackedEnsembleEstimator(base_models=bm, **self.hyper_parameters)
            else:
                if self.meta_learner!=None:
                    temp=self.model_dict[self.meta_learner]
                    if(meta_learner_params!=None):
                        temp.set_params(**meta_learner_params)
                    self.model=ensemble.StackingClassifier(estimators=self.estemators,final_estimator=temp,**self.hyper_parameters)
                else:
                    self.model=ensemble.StackingClassifier(estimators=self.estemators,**self.hyper_parameters)
        self.models={self.estemators[i][0]:self.estemators[i][1] for i in range(len(self.estemators))}
        self.models["Ensamble"]=self.model

 