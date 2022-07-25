
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

warnings.filterwarnings("ignore")
sys.path.insert(0, '..')
import EvaluationModule 
from ModelModule.models import make_models


def test_hyper_parameters(parameters, train,test,dataset, aml):
    grid=list(model_selection.ParameterGrid(parameters))
    for params in tqdm.tqdm(range(len(grid))):
        model=make_ensamble(**grid[params]).model
        aml.validate(str(grid[params]),model,train,test)
        aml.evaluate(str(grid[params]),model,test)
    aml.validation_output(dataset=dataset)
    aml.test_output(dataset=dataset)
class make_ensamble(make_models): 
    def __init__(self,model_id, model_list,base_model_params=[], ensamble_type='Stacking', meta_learner=None, meta_learner_params=None, hyper_parameters=None, challange="Regression"):
        ## exception cehcking. 
        make_models.__init__(self,model_id,model_list,base_model_params,challange)
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
        if self.challange=='Regression':
            if self.ensamble_type=="Bagging":
                self.model=ensemble.BaggingRegressor(base_estimator=self.estemators[0][1],**self.hyper_parameters)
            elif self.ensamble_type=="Boosting":
                self.model=ensemble.AdaBoostRegressor(base_estimator=self.estemators[0][1],**self.hyper_parameters)
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
            else:
                if self.meta_learner!=None:
                    temp=self.model_dict[self.meta_learner]
                    if(meta_learner_params!=None):
                        temp.set_params(**meta_learner_params)
                    self.model=ensemble.StackingClassifier(estimators=self.estemators,final_estimator=temp,**self.hyper_parameters)
                else:
                    self.model=ensemble.StackingClassifier(estimators=self.estemators,**self.hyper_parameters)
        self.models=b={self.estemators[i][0]:self.estemators[i][1] for i in range(len(self.estemators))}
        self.models["Ensamble"]=self.model