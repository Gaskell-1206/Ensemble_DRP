import csv
import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import smogn
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.model_selection import (KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, ShuffleSplit,
                                     StratifiedKFold)


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
    return metrics.f1_score(true, pred, average='macro')


def confusion_matrix_scratch(true, pred, normalize, plot):
    if normalize:
        contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=[
            'prediction'], normalize=True)
    else:
        contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=[
            'prediction'], normalize=False)
    if plot:
        sns.heatmap(contingency_matrix.T, annot=True,
                    fmt='.2f', cmap="YlGnBu", cbar=False)
    else:
        return contingency_matrix


class AutoBuild():
    def __init__(self, seed=1, project_name="EHR_RA_SC", challenge="regression", balance_class=0):
        self.seed = seed
        self.project_name = project_name
        self.challenge = challenge
        if challenge == "regression":
            self.target = "DAS28_CRP_3M"
        elif challenge == "regression_delta" or challenge == "regression_delta_binary":
            self.target = "delta"
        elif challenge == "classification":
            self.target = "DrugResponse"
        elif challenge == "binary_classification":
            self.target = "DrugResponse_binary"
        self.balance_class = balance_class

        self.train_perf = pd.DataFrame(columns=[
                                       "model", "MAE", "MSE", "RMSE", "R2", "Pearson_Correlation", "Accuracy", "F1-Score"])
        self.val_perf = pd.DataFrame(columns=[
            "model", "MAE", "MSE", "RMSE", "R2", "Pearson_Correlation", "Accuracy", "F1-Score"])
        self.test_perf = pd.DataFrame(columns=[
            "model", "MAE", "MSE", "RMSE", "R2", "Pearson_Correlation", "Accuracy", "F1-Score"])
        self.regression_leaderboard = pd.DataFrame(
            columns=["model", "MAE", "MSE", "RMSE", "R2", "Pearson_Correlation"])
        self.classification_leaderboard = pd.DataFrame(
            columns=["model", "Accuracy", "F1-Score"])
        self.saved_model = {}
        self.best_model = ''

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

    def responseClassify_binary(self, row, baseline, next):
        # set threshold
        lower_change = 0.6
        upper_change = 1.2

        if "delta" in self.challenge:
            change = row[next]
            row[next] = row[baseline] - change
        else:
            change = row[baseline] - row[next]

        if change <= lower_change:
            return "Nonresponder"

        elif (change <= upper_change) & (change > lower_change):
            if row[next] > 5.1:
                return "Nonresponder"
            else:
                return "Responder"

        elif change > upper_change:
            if row[next] > 3.2:
                return "Responder"
            else:
                return "Responder"

        else:
            return 2

    def validate(self, model_id, estimator, trainset, testset):
        X = trainset.iloc[:, :-1]
        y = trainset.iloc[:, -1]
        balance_class = self.balance_class

        if "regression" in self.challenge:
            # define balance class pipeline
            if balance_class == 2:
                # create pseudo-class for StratifiedKFold
                if "binary" in self.challenge:
                    X.loc[:, self.target] = y
                    y = X.apply(lambda row: self.responseClassify_binary(
                        row, 'DAS28_CRP_0M', self.target), axis=1)
                elif self.challenge == 'regression':
                    X.loc[:, self.target] = y
                    y = X.apply(lambda row: self.responseClassify(
                        row, 'DAS28_CRP_0M', self.target), axis=1)
                cv = RepeatedStratifiedKFold(
                    n_splits=10, n_repeats=3, random_state=self.seed)
                # cv = KFold(n_splits=10, shuffle=True, random_state=self.seed)
            elif balance_class == 1:
                # # create pseudo-class for StratifiedKFold
                if "binary" in self.challenge:
                    X.loc[:, self.target] = y
                    y = X.apply(lambda row: self.responseClassify_binary(
                        row, 'DAS28_CRP_0M', self.target), axis=1)
                else:
                    X.loc[:, self.target] = y
                    y = X.apply(lambda row: self.responseClassify(
                        row, 'DAS28_CRP_0M', self.target), axis=1)
                # print("train y:", y.value_counts())
                cv = RepeatedStratifiedKFold(
                    n_splits=10, n_repeats=3, random_state=self.seed)
                # cv = KFold(n_splits=10, shuffle=True, random_state=self.seed)

            else:
                cv = RepeatedKFold(n_splits=10, n_repeats=3,
                                   shuffle=True, random_state=self.seed)

            cv = cv.split(X, y)

            for train_index, test_index in cv:
                if balance_class == 2:
                    X_train = X.iloc[train_index, :]
                    y_train = y.iloc[train_index]
                    X_val = X.iloc[test_index, :-1]
                    y_val = X.iloc[test_index, -1]
                    print("before sampling:", y_train.value_counts())
                    # print("class y_val:",y.iloc[test_index].value_counts())
                    resample = SMOTEENN(enn=EditedNearestNeighbours(
                        sampling_strategy='auto', n_neighbors=3))
                    X_train, y_train = resample.fit_resample(X_train, y_train)
                    print("after sampling:", y_train.value_counts())
                    X_train = X_train.iloc[:, :-1]
                    y_train = X_train.iloc[:, -1]
                    # print("after sampling:",y_train)

                elif balance_class == 1:
                    X_train = X.iloc[train_index, :-1]
                    y_train = y.iloc[train_index]
                    X_val = X.iloc[test_index, :-1]
                    y_val = y.iloc[test_index]
                    data_for_balance = X.iloc[train_index, :].reset_index(
                        drop=True)
                    try:
                        train = smogn.smoter(
                            data=data_for_balance, y=self.target, samp_method="balance")
                    except ValueError as e:
                        # print(e)
                        pass
                    X_train = train.iloc[:, :-1]
                    y_train = train.iloc[:, -1]
                    X_val = X.iloc[test_index, :-1]
                    y_val = X.iloc[test_index, -1]

                else:
                    X_train, X_val = X.iloc[train_index,
                                            :], X.iloc[test_index, :]
                    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

                # summarize train and test composition
                estimator.fit(X_train, y_train)

                working_set = X_train
                working_set[self.target] = y_train.values
                self.evaluate(model_id, estimator, "train",
                              working_set, self.train_perf)
                working_set = X_val
                working_set[self.target] = y_val.values
                self.evaluate(model_id, estimator, "val",
                              working_set, self.val_perf)

        elif "classification" in self.challenge:
            cv = RepeatedStratifiedKFold(
                n_splits=10, n_repeats=3, random_state=self.seed)
            cv = cv.split(X, y)

            for train_index, test_index in cv:
                X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_val = y.iloc[train_index], y.iloc[test_index]

                # define balance class pipeline
                resample = SMOTETomek(
                    tomek=TomekLinks(sampling_strategy='auto'))
                X_train, y_train = resample.fit_resample(X_train, y_train)
                # print("after balancing class:", y_train.value_counts())
                estimator.fit(X_train, y_train)

                working_set = X_train
                working_set[self.target] = y_train.values
                self.evaluate(model_id, estimator, "train",
                              working_set, self.train_perf)
                working_set = X_val
                working_set[self.target] = y_val.values
                self.evaluate(model_id, estimator, "val",
                              working_set, self.val_perf)

        # use all data in trainset to train model and testset to evaluate performance
        X = trainset.iloc[:, :-1]
        y = trainset.iloc[:, -1]
        estimator.fit(X, y)
        self.evaluate(model_id, estimator, "test", testset, self.test_perf)

    def evaluate(self, model_id, model, dataset, working_set, save_df):
        X = working_set.drop(columns=self.target)
        true = working_set[self.target]
        pred = model.predict(X)
        baseline = working_set['DAS28_CRP_0M']

        baseline, true, pred = np.array(
            baseline), np.array(true), np.array(pred)
        assert len(baseline) == len(true)
        assert len(true) == len(pred)

        df = pd.DataFrame(list(zip(baseline, true, pred)),
                          columns=['baseline', 'true', 'pred'])

        if "regression" in self.challenge:

            df = pd.DataFrame(list(zip(baseline, true, pred)), columns=[
                'baseline', 'true', 'pred'])

            if "binary" in self.challenge:
                # get classification target
                classification_true = df.apply(
                    lambda row: self.responseClassify_binary(row, 'baseline', 'true'), axis=1)
                classification_pred = df.apply(
                    lambda row: self.responseClassify_binary(row, 'baseline', 'pred'), axis=1)
                self.saved_model[model_id] = (
                    classification_true, classification_pred)
            else:
                # get classification target
                classification_true = df.apply(
                    lambda row: self.responseClassify(row, 'baseline', 'true'), axis=1)
                classification_pred = df.apply(
                    lambda row: self.responseClassify(row, 'baseline', 'pred'), axis=1)
                self.saved_model[model_id] = (
                    classification_true, classification_pred)

            save_df.loc[len(save_df.index)] = [model_id,
                                               MAE(true,
                                                   pred),
                                               MSE(true,
                                                   pred),
                                               RMSE(
                                                   true, pred),
                                               R2(true,
                                                  pred),
                                               Pearson_Correlation(
                                                   true, pred),
                                               Classification_Accuracy(
                                                   classification_true, classification_pred),
                                               F1_Score(classification_true, classification_pred)]

        elif "classification" in self.challenge:
            save_df.loc[len(save_df.index)] = [model_id,
                                               None,
                                               None,
                                               None,
                                               None,
                                               None,
                                               Classification_Accuracy(
                                                   true, pred),
                                               F1_Score(true, pred)]
            if dataset == "test":
                self.saved_model[model_id] = (df['true'], df['pred'])

    def evaluate_explore(self, model_id, model, test):
        X_test = test.drop(columns=self.target)
        true = test[self.target]
        pred = model.predict(X_test)
        baseline = test['DAS28_CRP_0M']

        baseline, true, pred = np.array(
            baseline), np.array(true), np.array(pred)
        assert len(baseline) == len(true)
        assert len(true) == len(pred)

        df = pd.DataFrame(list(zip(baseline, true, pred)),
                          columns=['baseline', 'true', 'pred'])

        if "regression" in self.challenge:
            self.regression_leaderboard.loc[len(self.regression_leaderboard.index)] = [model_id,
                                                                                       MAE(true,
                                                                                           pred),
                                                                                       MSE(true,
                                                                                           pred),
                                                                                       RMSE(
                                                                                           true, pred),
                                                                                       R2(true,
                                                                                          pred),
                                                                                       Pearson_Correlation(true, pred)]

            if "binary" in self.challenge:
                # get classification target
                classification_true = df.apply(
                    lambda row: self.responseClassify_binary(row, 'baseline', 'true'), axis=1)
                classification_pred = df.apply(
                    lambda row: self.responseClassify_binary(row, 'baseline', 'pred'), axis=1)
                self.saved_model[model_id] = (
                    classification_true, classification_pred)
            else:
                # get classification target
                classification_true = df.apply(
                    lambda row: self.responseClassify(row, 'baseline', 'true'), axis=1)
                classification_pred = df.apply(
                    lambda row: self.responseClassify(row, 'baseline', 'pred'), axis=1)
                self.saved_model[model_id] = (
                    classification_true, classification_pred)

            self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [model_id,
                                                                                               Classification_Accuracy(
                                                                                                   classification_true, classification_pred),
                                                                                               F1_Score(classification_true, classification_pred)]

        elif "classification" in self.challenge:
            self.classification_leaderboard.loc[len(self.classification_leaderboard.index)] = [model_id,
                                                                                               Classification_Accuracy(
                                                                                                   df['true'], df['pred']),
                                                                                               F1_Score(df['true'], df['pred'])]
            self.saved_model[model_id] = (df['true'], df['pred'])

    def confusion_matrix(self, model_id, plot=True, normalize=True):
        true, pred = self.saved_model[model_id]
        print(pred.value_counts())
        if normalize:
            contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=[
                                             'prediction'], normalize=True)
        else:
            contingency_matrix = pd.crosstab(true, pred, rownames=['true'], colnames=[
                                             'prediction'], normalize=False)
        if plot:
            sns.heatmap(contingency_matrix.T, annot=True,
                        fmt='.2f', cmap="YlGnBu", cbar=False)
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

    def validation_output(self, dataset, output='../leaderboard/'):
        validation = self.val_perf
        path = os.path.join(output, f'{self.project_name}_output.csv')
        header = ["dataset", "challenge", "process_approach", "imputation", "patient_group", "drug_group", "train_test_rate", "remove_low_DAS", "random_state",
                  "model_id", "MAE", "MSE", "RMSE", 'R2', "Pearson_Correlation", "Accuracy", "F1-Score"]

        # if header exists
        has_header = False
        try:
            if len(pd.read_csv(path)) > 0:
                has_header = True
        except:
            has_header = False

        # save validation output to csv
        with open(path, 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            if not has_header:
                # write the header
                writer.writerow(header)
            for index, rows in validation.iterrows():
                data = ["validation", dataset.challenge, dataset.process_approach, dataset.imputation, dataset.patient_group, dataset.drug_group, dataset.train_test_rate, dataset.remove_low_DAS, dataset.random_state,
                        rows["model"], rows["MAE"], rows["MSE"], rows["RMSE"], rows["R2"], rows["Pearson_Correlation"], rows["Accuracy"], rows["F1-Score"]]
                # write the data
                writer.writerow(data)

    def test_output(self, dataset, output='../leaderboard/'):
        test = self.test_perf
        path = os.path.join(output, f'{self.project_name}_output.csv')

        header = ["dataset", "challenge", "process_approach", "imputation", "patient_group", "drug_group", "train_test_rate", "remove_low_DAS", "random_state",
                  "model_id", "MAE", "MSE", "RMSE", 'R2', "Pearson_Correlation", "Accuracy", "F1-Score"]

        # if header exists
        has_header = False
        try:
            if len(pd.read_csv(path)) > 0:
                has_header = True
        except:
            has_header = False

        # save validation output to csv
        with open(path, 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            if not has_header:
                # write the header
                writer.writerow(header)
            for index, rows in test.iterrows():
                data = ["test", dataset.challenge, dataset.process_approach, dataset.imputation, dataset.patient_group, dataset.drug_group, dataset.train_test_rate, dataset.remove_low_DAS, dataset.random_state,
                        rows["model"], rows["MAE"], rows["MSE"], rows["RMSE"], rows["R2"], rows["Pearson_Correlation"], rows["Accuracy"], rows["F1-Score"]]
                # write the data
                writer.writerow(data)
