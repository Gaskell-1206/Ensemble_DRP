from pandas.api.types import is_dict_like
import sklearn as sk
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import ensemble
class ensamble_model: 
    def __init__(self, model_list, ensamble_type='Stacking', meta_learner=None, meta_learner_params=None, hyper_parameters=None):
        ## exception cehcking. 
        if is_dict_like(model_list)==False:
            raise ValueError("model_list must be dict like")
        if is_dict_like(hyper_parameters)==False and hyper_parameters!=None:
            raise ValueError("hyper_parameters must be dict like")
        for model in model_list:
            if model not in ('KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes", 'ANN'):
                raise ValueError("model must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes", 'ANN')
        if meta_learner not in ('KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes",'ANN', None):
            raise ValueError("meta_learner must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso','ANN', "SVM", "Bayes")
        if ensamble_type not in ('Stacking', 'Bagging', 'Boosting'):
            raise ValueError("ensamble_type must be in ", 'Stacking', 'Bagging', 'Boosting')
        if len(model_list)!=1 and ensamble_type in ( 'Bagging', 'Boosting'):
            raise ValueError(ensamble_type, 'requires a homogeneous weak learner ')

        ## intitlizing atrivutes    
        self.base_model_list=model_list
        self.n_weak_learners=len(model_list)
        self.ensamble_type=ensamble_type
        self.meta_learner=meta_learner
        
        if hyper_parameters==None:
            hyper_parameters=dict()
        self.hyper_parameters=hyper_parameters
        

        ## dictionary for base models
        models = dict()
        models['Linear'] = sk.linear_model.LinearRegression()
        models['ANN'] = neural_network.MLPRegressor()
        models['Ridge'] = sk.linear_model.Ridge() 
        models['KNN'] = sk.neighbors.KNeighborsRegressor() 
        models['Bayes'] = naive_bayes.GaussianNB 
        models['Logistic'] = sk.linear_model.LogisticRegression() 
        models['Lasso'] = sk.linear_model.Lasso() 
        models['SVM'] = sk.svm.SVR()

        self.model_dict=models
        # list of tuples for stacking  
        estemator=[]
        for model in self.base_model_list:
            temp=self.model_dict[model]
            temp.set_params(**self.base_model_list[model])
            estemator.append((model,temp))
        self.estemators=estemator 
        ## difinging model type
        if self.ensamble_type=="Bagging":
            self.model=ensemble.BaggingRegressor(base_estimator=self.estemators[0][1],**self.hyper_parameters)
        elif self.ensamble_type=="Boosting":
            self.model=ensemble.AdaBoostRegressor(base_estimator=self.estemators[0][1],**self.hyper_parameters)
        else:
            temp=self.model_dict[self.meta_learner]
            temp.set_params(**meta_learner_params)
            self.model=ensemble.StackingRegressor(estimators=self.estemators,final_estimator=temp,**self.hyper_parameters)
            
''''
todo: 
- fix it so that mutliple of the same model can be added, this will likely come down to naming. 
'''
