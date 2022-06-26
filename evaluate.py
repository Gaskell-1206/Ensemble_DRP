import pathlib
import numpy as np
import pandas as pd
import csv
from sklearn import metrics
from scipy.stats import pearsonr
from DataModule.Data_Preparation import responseClassify


def R2(true, pred):
    return metrics.r2_score(true, pred)

def AE(true, pred):
    return np.abs(np.array(true)-np.array(pred))

def SE(true, pred):
    return np.power(np.array(true)-np.array(pred), 2)

def MAE(true, pred):
    return np.mean(AE(true, pred))

def MSE(true, pred):
    return np.mean(SE(true, pred))

def RMSE(true, pred):
    return np.sqrt(MSE(true, pred))

def Pearson_Correlation(true, pred):
    return pearsonr(true, pred)[0]
    
def Classification_Accuracy(true, pred):
    return (sum([1.0 for x,y in zip(true,pred) if x == y]) /
            len(true))
    
class AutoBuild():
    
    def __init__(self, seed=1, project_name = "EHR_RA_SC"):
        self.seed = seed
        self.project_name = project_name
        self.regression_leaderboard = pd.DataFrame(columns=["model","MAE","MSE","RMSE","R2","Pearson_Correlation"])
        self.classification_leaderboard = pd.DataFrame(columns=["model","Accuracy"])
    
    def evaluate(self, algorithm, baseline, true, pred):
        baseline, true, pred = np.array(baseline), np.array(true), np.array(pred)
        assert len(baseline) == len(true)
        assert len(true) == len(pred)
        df = pd.DataFrame(list(zip(baseline,true,pred)),columns=['baseline','true','pred'])
        
        self.regression_leaderboard.loc[len(self.regression_leaderboard.index)] = [algorithm,
                                                                                   MAE(true,pred),
                                                                                   MSE(true,pred),
                                                                                   RMSE(true, pred),
                                                                                   R2(true,pred),
                                                                                   Pearson_Correlation(true, pred)]
        
        # get classification target
        classification_true = df.apply(lambda row: responseClassify(row,'baseline','true'), axis=1)
        classification_pred = df.apply(lambda row: responseClassify(row,'baseline','pred'), axis=1)
        
        self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [algorithm,
                                                                                           Classification_Accuracy(classification_true,classification_pred)]
        
    def leaderboard(self):
        return self.regression_leaderboard, self.classification_leaderboard