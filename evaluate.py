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

class AutoBuild():
    def __init__(self, seed=1, project_name="EHR_RA_SC", challenge="regression"):
        self.seed = seed
        self.project_name = project_name
        self.challenge = challenge
        if challenge == "regression":
            self.target = "DAS28_CRP_3M"
        elif challenge == "regression_delta":
            self.target = "delta"
        elif challenge == "classification":
            self.target = "DrugResponse"
        elif challenge == "binary_classification":
            self.target = "DrugResponse_binary"
        self.regression_leaderboard = pd.DataFrame(
            columns=["model", "MAE", "MSE", "RMSE", "R2", "Pearson_Correlation"])
        self.classification_leaderboard = pd.DataFrame(
            columns=["model", "Accuracy","F1-Score"])
        self.saved_model = {}
        
    def responseClassify(self, row, baseline, next):
        # set threshold
        lower_change = 0.6
        upper_change = 1.2
        
        if self.challenge == "regression_delta":
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

    def evaluate(self, model_id, model, test):
        X_test = test.drop(columns=self.target)
        true = test[self.target]
        pred = model.predict(X_test)
        baseline = test['DAS28_CRP_0M']
            
        baseline, true, pred = np.array(baseline), np.array(true), np.array(pred)
        assert len(baseline) == len(true)
        assert len(true) == len(pred)
        
        df = pd.DataFrame(list(zip(baseline, true, pred)), columns=['baseline', 'true', 'pred'])
        
        if "regression" in self.challenge:

            self.regression_leaderboard.loc[len(self.regression_leaderboard.index)] = [model_id,
                                                                                    MAE(true,pred),
                                                                                    MSE(true,pred),
                                                                                    RMSE(true, pred),
                                                                                    R2(true,pred),
                                                                                    Pearson_Correlation(true, pred)]
            
            # get classification target
            classification_true = df.apply(
                lambda row: self.responseClassify(row, 'baseline', 'true'), axis=1)
            classification_pred = df.apply(
                lambda row: self.responseClassify(row, 'baseline', 'pred'), axis=1)
            self.saved_model[model_id] = (classification_true, classification_pred)
            
            self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [model_id,
                                                                                           Classification_Accuracy(classification_true, classification_pred),
                                                                                           F1_Score(classification_true, classification_pred)]
            
        elif "classification" in self.challenge:
            self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [model_id,
                                                                                            Classification_Accuracy(df['true'], df['pred']),
                                                                                            F1_Score(df['true'], df['pred'])]
            self.saved_model[model_id] = (df['true'], df['pred'])
        
    def leaderboard(self):
        if "regression" in self.challenge:
            return self.regression_leaderboard, self.classification_leaderboard
        elif "classification" in self.challenge:
            return None, self.classification_leaderboard
        else:
            print("challenge does not exist")

    def confusion_matrix(self, model_id, plot=True, normalize=True):
        true, pred = self.saved_model[model_id]
        if normalize:
            contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=['prediction'],normalize=True)
        else:
            contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=['prediction'],normalize=False)
        if plot:
            sns.heatmap(contingency_matrix.T, annot=True, fmt='.2f', cmap="YlGnBu", cbar=False)
        else:
            return contingency_matrix

    def plot_results(self, mode, metics):
        if mode == 'classification':
            results = self.classification_leaderboard

            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.barh(results['model'], results[metics])
            ax.set_xlabel(metics)
            ax.set_ylabel('Models')
            ax.set_title(f'Classification {metics}')
            # ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
            # ax.set_yticks(np.arange(0, 81, 10))
            # ax.legend(labels=['Men', 'Women'])
            plt.show()

        elif mode == 'regression':
            results = self.regression_leaderboard
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.barh(results['model'], results[metics])
            ax.set_xlabel(metics)
            ax.set_ylabel('Models')
            ax.set_title(f'Regression {metics}')
            # ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
            # ax.set_yticks(np.arange(0, 81, 10))
            # ax.legend(labels=['Men', 'Women'])
            plt.show()
            
        else:
            print("mode does not exist")