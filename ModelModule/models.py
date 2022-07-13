import sklearn as sk
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree 
from sklearn import model_selection
import xgboost as xgb
class make_models():
    def __init__(self,model_id,model_list,base_model_params=[],challange="Regression"):
        if challange not in ('Regression', 'Classification'):
            raise ValueError(challange,'Challange must be either Regression or Classification')
        for model in model_list:
            if model not in ('KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", 'ANN','XGBoost','Random Forest', 'Tree'):
                raise ValueError(model, "model must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes", 'ANN','XGBoost','Random Forest','Tree' )
        temp=dict()
        for i in range(len(model_list)):
            if(i>=len(base_model_params)):
                base_model_params.append({})
            temp[model_list[i]]=base_model_params[i]
        model_list=temp
        ## intitlizing atrivutes    
        self.base_model_list=model_list
        self.model_id=model_id
        self.challange=challange
        
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
        