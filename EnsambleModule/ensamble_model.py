from pandas.api.types import is_dict_like
import sklearn as sk
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree 
import xgboost as xgb
class make_ensamble: 
    def __init__(self, model_list, ensamble_type='Stacking', meta_learner=None, meta_learner_params=None, hyper_parameters=None, challange="Regression"):
        ## exception cehcking. 
        if challange not in ('Regression', 'Classification'):
            raise ValueError('Challange must be either Regression or Classification')
        if is_dict_like(model_list)==False:
            raise ValueError("model_list must be dict like")
        if is_dict_like(hyper_parameters)==False and hyper_parameters!=None:
            raise ValueError("hyper_parameters must be dict like")
        for model in model_list:
            if model not in ('KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes", 'ANN','XGBoost','Random Forest', 'Tree'):
                raise ValueError("model must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes", 'ANN','XGBoost','Random Forest','Tree' )
        if meta_learner not in ( 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes",'ANN','XGBoost','Random Forest','Tree', None ):
            raise ValueError("meta_learner must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso','ANN', "SVM", "Bayes",'XGBoost','Random Forest','Tree')
        if ensamble_type not in ('Stacking', 'Bagging', 'Boosting'):
            raise ValueError("ensamble_type must be in ", 'Stacking', 'Bagging', 'Boosting')
        if len(model_list)!=1 and ensamble_type in ( 'Bagging', 'Boosting'):
            raise ValueError(ensamble_type, 'requires a homogeneous weak learner ')

        ## intitlizing atrivutes    
        self.challange=challange
        self.base_model_list=model_list
        self.n_weak_learners=len(model_list)
        self.ensamble_type=ensamble_type
        self.meta_learner=meta_learner
        
        if hyper_parameters==None:
            hyper_parameters=dict()
        self.hyper_parameters=hyper_parameters
        

        ## dictionary for base models split between classfication and regression. 


        models = dict()
        if(self.challange=='Regression'):
            models['Linear'] = sk.linear_model.LinearRegression()
            models['ANN'] = neural_network.MLPRegressor()
            models['Ridge'] = sk.linear_model.Ridge() 
            models['KNN'] = sk.neighbors.KNeighborsRegressor() 
            models['Lasso'] = sk.linear_model.Lasso() 
            models['SVM'] = sk.svm.SVR()
            models['Random Forest']=ensemble.RandomForestRegressor()
            models['Tree']=tree.DecisionTreeRegressor()
            models['XGBoost']=xgb.XGBRegressor()
        else: 
            models['Bayes'] = naive_bayes.GaussianNB ()
            models['Logistic'] = sk.linear_model.LogisticRegression()
            models['ANN'] = neural_network.MLPClassifier()
            models['KNN'] = sk.neighbors.KNeighborsClassifier()
            models['SVM'] = sk.svm.SVC()
            models['Random Forest']=ensemble.RandomForestClassifier()
            models['XGBoost']=xgb.XGBClassifier()
            models['Tree']=tree.DecisionTreeClassifier()



        self.model_dict=models
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
                    temp.set_params(**meta_learner_params)
                    self.model=ensemble.StackingClassifier(estimators=self.estemators,final_estimator=temp,**self.hyper_parameters)
                else:
                    self.model=ensemble.StackingClassifier(estimators=self.estemators,**self.hyper_parameters)
        self.models=b={self.estemators[i][0]:self.estemators[i][1] for i in range(len(self.estemators))}
        self.models["Ensamble"]=self.model



''''
todo: 
- fix it so that mutliple of the same model can be added, this will likely come down to naming. 
'''
