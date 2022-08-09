import sklearn as sk
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree 
from sklearn import model_selection
import xgboost as xgb
import pickle
import h2o
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.automl import H2OAutoML
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators import H2OGradientBoostingEstimator
class model_primitive():
    def __init__(self,model_list,base_model_params=[],model_id="",challenge="Regression", h20=True):
        if challenge not in ('Regression', 'Classification'):
            raise ValueError(challenge,'Challange must be either Regression or Classification')
        if(h20):
            h2o.init()
        self.model_id=model_id
        self.h20=h20
        self.challenge=challenge
        models = dict()
        if(self.challenge=='Regression'):
            models['AutoML']=H2OAutoML()
            models["Deep Learning"]=H2ODeepLearningEstimator()
            models['ANN'] = neural_network.MLPRegressor()
            models['Ridge'] = sk.linear_model.Ridge() 
            models['KNN'] = sk.neighbors.KNeighborsRegressor() 
            models['Lasso'] = sk.linear_model.Lasso() 
            models['GBM'] = H2OGradientBoostingEstimator()
            if(h20):
                models['Random Forest']=H2ORandomForestEstimator()
                models['SVM'] = H2OSupportVectorMachineEstimator()
                models['XGBoost']=H2OXGBoostEstimator()
                models['Linear'] = H2OGeneralizedLinearEstimator()
            else:
                models['Random Forest']=ensemble.RandomForestRegressor()
                models['SVM'] = sk.svm.SVR()
                models['XGBoost']=xgb.XGBRegressor()
                models['Linear'] = sk.linear_model.LinearRegression()
            
            models['Tree']=tree.DecisionTreeRegressor()
            
        else:
            models['AutoML']=H2OAutoML() 
            models['GBM'] = H2OGradientBoostingEstimator()
            models["Deep Learning"]=H2ODeepLearningEstimator()
            models['ANN'] = neural_network.MLPClassifier()
            models['KNN'] = sk.neighbors.KNeighborsClassifier()
            #if(h20):
            models['Random Forest']=H2ORandomForestEstimator()
            models['SVM'] = H2OSupportVectorMachineEstimator()
            models['XGBoost']=H2OXGBoostEstimator()
            models['Linear'] = H2OGeneralizedLinearEstimator()
            models['Bayes'] = H2ONaiveBayesEstimator()
            # else:
            #     models['Logistic'] = sk.linear_model.LogisticRegression()
            #     models['SVM'] = sk.svm.SVR()
            #     models['Bayes'] = naive_bayes.GaussianNB ()
            #     models['Random Forest']=ensemble.RandomForestClassifier()
            #     models['XGBoost']=xgb.XGBClassifier()
            models['Tree']=tree.DecisionTreeClassifier()
        self.model_dict=models