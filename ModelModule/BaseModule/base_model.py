
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
        print(grid[params])
        model=make_model(model_id=str(grid[params]), model_list=grid[params]["model_list"]).model
        del grid[params]["model_list"]
        model.set_params(**grid[params])
        aml.validate(str(grid[params]),model,train,test)
        aml.evaluate(str(grid[params]),model,test)
    aml.validation_output(dataset=dataset)
    aml.test_output(dataset=dataset)

class make_model(make_models):
    def __init__(self,model_id,model_list,base_model_params=[],challange="Regression"):
        make_models.__init__(self,model_id,model_list,base_model_params,challange="Regression")
        if(len(self.base_model_list)!=1):
            raise ValueError("lenght of model list is ", len(self.base_model_list), "not 1")
        for model in self.base_model_list:
            temp=self.model_dict[model]
            temp.set_params(**self.base_model_list[model])
        self.model=temp
