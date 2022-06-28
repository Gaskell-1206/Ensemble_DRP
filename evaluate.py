import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
from sklearn import metrics
from scipy.stats import pearsonr


def R2(true, pred):
    return metrics.r2_score(true, pred)


def MAE(true, pred):
    return metrics.mean_absolute_error(true, pred)


def MSE(true, pred):
    return metrics.mean_squared_error(true, pred)


def RMSE(true, pred):
    return metrics.mean_squared_error(true, pred, squared=False)


def Pearson_Correlation(true, pred):
    return pearsonr(true, pred)[0]


def Classification_Accuracy(true, pred):
    return metrics.accuracy_score(true, pred)


def F1_Score(true, pred):
    return metrics.f1_score(true, pred,average='macro')

def responseClassify(row, baseline, next, delta=False):
    # set threshold
    lower_change = 0.6
    upper_change = 1.2
    
    if delta:
        change = row[next]
        row[next] = row[baseline] - change
    else:
        change = row[baseline] - row[next]

    if change <= lower_change:
        return "No Response"

    elif (change <= upper_change) & (change > lower_change):
        if row[next] > 5.1:
            return "No Response"
        else:
            return "Moderate"

    elif change > upper_change:
        if row[next] > 3.2:
            return "Moderate"
        else:
            return "Good"

    else:
        return "Unknown"


class AutoBuild():
    def __init__(self, seed=1, project_name="EHR_RA_SC", challenge="two_stage"):
        self.seed = seed
        self.project_name = project_name
        self.challenge = challenge
        self.regression_leaderboard = pd.DataFrame(
            columns=["model", "MAE", "MSE", "RMSE", "R2", "Pearson_Correlation"])
        self.classification_leaderboard = pd.DataFrame(
            columns=["model", "Accuracy","F1-Score"])
        self.saved_model = {}

    def evaluate(self, model_name, true, pred):
        baseline = true['DAS28_CRP_0M']
        true = true['delta'] if self.challenge == 'regression' else true['DAS28_CRP_3M']
        baseline, true, pred = np.array(
            baseline), np.array(true), np.array(pred)
        assert len(baseline) == len(true)
        assert len(true) == len(pred)
        
        if self.challenge == "regression":
            df = pd.DataFrame(list(zip(baseline, true, pred)),
                            columns=['baseline', 'true', 'pred'])

            self.regression_leaderboard.loc[len(self.regression_leaderboard.index)] = [model_name,
                                                                                    MAE(true,pred),
                                                                                    MSE(true,pred),
                                                                                    RMSE(true, pred),
                                                                                    R2(true,pred),
                                                                                    Pearson_Correlation(true, pred)]
            
            # get classification target
            classification_true = df.apply(
                lambda row: responseClassify(row, 'baseline', 'true', True), axis=1)
            classification_pred = df.apply(
                lambda row: responseClassify(row, 'baseline', 'pred', True), axis=1)
            self.saved_model[model_name] = (
                classification_true, classification_pred)
            
            self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [model_name,
                                                                                           Classification_Accuracy(classification_true, classification_pred),
                                                                                           F1_Score(classification_true, classification_pred)]
            
        elif self.challenge == "classification":
            self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [model_name,
                                                                                            Classification_Accuracy(true, pred),
                                                                                            F1_Score(true, pred)]
        
        elif self.challenge == "two_stage":
            df = pd.DataFrame(list(zip(baseline, true, pred)),
                            columns=['baseline', 'true', 'pred'])

            self.regression_leaderboard.loc[len(self.regression_leaderboard.index)] = [model_name,
                                                                                    MAE(true,pred),
                                                                                    MSE(true,pred),
                                                                                    RMSE(true, pred),
                                                                                    R2(true,pred),
                                                                                    Pearson_Correlation(true, pred)]
            
            # get classification target
            classification_true = df.apply(
                lambda row: responseClassify(row, 'baseline', 'true', False), axis=1)
            classification_pred = df.apply(
                lambda row: responseClassify(row, 'baseline', 'pred', False), axis=1)
            self.saved_model[model_name] = (
                classification_true, classification_pred)
            
            self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [model_name,
                                                                                           Classification_Accuracy(classification_true, classification_pred),
                                                                                           F1_Score(classification_true, classification_pred)]

    def leaderboard(self):
        if self.challenge == "regression":
            return self.regression_leaderboard, self.classification_leaderboard
        elif self.challenge == "classification":
            return self.classification_leaderboard
        elif self.challenge == "two_stage":
            return self.regression_leaderboard, self.classification_leaderboard

    def confusion_matrix(self, model_name, plot=True, normalize=True):
        true, pred = self.saved_model[model_name]
        # title_list = ['Good', 'Moderate', 'No Response']
        # matrix = metrics.confusion_matrix(true, pred)
        # matrix = pd.DataFrame(matrix, columns=title_list, index=title_list)
        if normalize:
            contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=['prediction'],normalize=True)
        else:
            contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=['prediction'],normalize=False)
        if plot:
            sns.heatmap(contingency_matrix.T, annot=True, fmt='.2f', cmap="YlGnBu", cbar=False)
        else:
            return contingency_matrix

    def plot_results(self, mode):
        if mode == 'classification':
            results = self.classification_leaderboard

            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.barh(results['model'], results['Accuracy'])
            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy')
            ax.set_title('Classification Accuracy')
            # ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
            # ax.set_yticks(np.arange(0, 81, 10))
            # ax.legend(labels=['Men', 'Women'])
            plt.show()

        elif mode == 'regression':
            results = self.regression_leaderboard
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.barh(results['model'], results['MSE'])
            ax.set_xlabel('Models')
            ax.set_ylabel('MSE')
            ax.set_title('Regression MSE')
            # ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
            # ax.set_yticks(np.arange(0, 81, 10))
            # ax.legend(labels=['Men', 'Women'])
            plt.show()
