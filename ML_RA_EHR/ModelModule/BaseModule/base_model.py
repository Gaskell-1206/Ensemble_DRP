
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
from ModelModule.model_primitive.model_primitive import model_primitive



def test_hyper_parameters(parameters, train,test,dataset, aml, name_list=[],output='../leaderboard/'):
    grid=list(model_selection.ParameterGrid(parameters))
    for params in (range(len(grid))):
        if(name_list==[]):
            model_is=str(grid[params])
        else:
            model_id=name_list[params]
        if "challenge" in list(grid[params].keys()):
           
            model=make_model(model_id=model_id, challenge=grid[params]["challenge"], model_list=grid[params]["model_list"]).model
            del grid[params]["challenge"]
        else:
        
           model=make_model(model_id=model_id, model_list=grid[params]["model_list"]).model
        
        del grid[params]["model_list"]
        model.set_params(**grid[params])
        aml.validate(model_id=model_id, estimator=model, trainset=train, testset=test)
        aml.evaluate(model_id=model_id,model=model,dataset="test",working_set=test, save_df= aml.train_perf  )
    # aml.validation_output(dataset=dataset,output=output)
    # aml.test_output(dataset=dataset,output=output)

class make_model(model_primitive):
    def __init__(self,model_id,model_list,base_model_params=[],challenge="Regression"):
        temp=dict()
        for i in range(len(model_list)):
            if(i>=len(base_model_params)):
                base_model_params.append({})
            temp[model_list[i]]=base_model_params[i]
        model_list=temp
        ## intitlizing atrivutes    
        self.base_model_list=model_list

        model_primitive.__init__(self,model_id,model_list,base_model_params,challenge)
        #if(len(self.base_model_list)!=1):
        #    raise ValueError("lenght of model list is ", len(self.base_model_list), "not 1")
        for model in self.base_model_list:

            temp=self.model_dict[model]
            temp.set_params(**self.base_model_list[model])
        self.model=temp
