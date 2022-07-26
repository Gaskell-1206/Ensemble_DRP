import sklearn as sk
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree 
from sklearn import model_selection
import xgboost as xgb
import pickle
class model_primitive():
    def __init__(self,model_list,base_model_params=[],model_id="",challenge="Regression"):
        if challenge not in ('Regression', 'Classification'):
            raise ValueError(challenge,'Challange must be either Regression or Classification')
        self.model_id=model_id
        self.challenge=challenge
        
        models = dict()
        if(self.challenge=='Regression'):
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