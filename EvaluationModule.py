import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import csv
import os
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit, cross_val_score
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
    return metrics.f1_score(true, pred, average='macro')


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
        self.validation = pd.DataFrame(columns=[
                                       "model", "MAE", "MSE", "RMSE", "R2", "Pearson_Correlation", "Accuracy", "F1-Score"])
        self.test = pd.DataFrame(columns=[
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

    # def validate(self, model_id, model, X_train, y_train, scoring):
    #     shuffle_split = ShuffleSplit(test_size=0.2, train_size=0.8, n_splits=10, random_state=self.seed)
    #     scores = cross_val_score(estimator=model, X=X_train,
    #                                y=y_train, scoring=scoring, cv=shuffle_split, verbose=0)

    #     self.validation.loc[len(self.validation.index)] = [model_id, metrics, scores]

    def validate(self, model_id, estimator, trainset, testset):
        X = trainset.iloc[:,:-1]
        y = trainset.iloc[:,-1]
        
        cv = ShuffleSplit(test_size=0.2, train_size=0.8,
                          n_splits=10, random_state=self.seed)
        cv = cv.split(X)

        if "regression" in self.challenge:
            for train_index, test_index in cv:
                X_train = X.iloc[train_index, :]
                X_val = X.iloc[test_index, :]
                y_train = y.iloc[train_index]
                y_val = y.iloc[test_index]
                baseline = X_val['DAS28_CRP_0M']

                estimator.fit(X_train, y_train)
                pred = estimator.predict(X_val)
                true = y_val

                df = pd.DataFrame(list(zip(baseline, true, pred)), columns=[
                                  'baseline', 'true', 'pred'])

                # get classification target
                classification_true = df.apply(
                    lambda row: self.responseClassify(row, 'baseline', 'true'), axis=1)
                classification_pred = df.apply(
                    lambda row: self.responseClassify(row, 'baseline', 'pred'), axis=1)
                # self.saved_model[model_id] = (
                #     classification_true, classification_pred)

                self.validation.loc[len(self.validation.index)] = [model_id,
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
            for train_index, test_index in cv:
                X_train = X.iloc[train_index, :]
                X_val = X.iloc[test_index, :]
                y_train = y.iloc[train_index]
                y_val = y.iloc[test_index]
                estimator.fit(X_train, y_train)
                pred = estimator.predict(X_val)
                true = y_val

                self.validation.loc[len(self.validation.index)] = [model_id,
                                                                   None,
                                                                   None,
                                                                   None,
                                                                   None,
                                                                   None,
                                                                   Classification_Accuracy(
                                                                       true, pred),
                                                                   F1_Score(true, pred)]
        
        # use testset to evaluate performance
        self.evaluate(model_id, estimator, testset)

    def evaluate(self, model_id, model, test):
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

            df = pd.DataFrame(list(zip(baseline, true, pred)), columns=[
                'baseline', 'true', 'pred'])

            # get classification target
            classification_true = df.apply(
                lambda row: self.responseClassify(row, 'baseline', 'true'), axis=1)
            classification_pred = df.apply(
                lambda row: self.responseClassify(row, 'baseline', 'pred'), axis=1)
            self.saved_model[model_id] = (
                classification_true, classification_pred)

            self.test.loc[len(self.validation.index)] = [model_id,
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
            self.test.loc[len(self.validation.index)] = [model_id,
                                                                None,
                                                                None,
                                                                None,
                                                                None,
                                                                None,
                                                                Classification_Accuracy(
                                                                    true, pred),
                                                                F1_Score(true, pred)]
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
        validation = self.validation
        path = os.path.join(output, f'{self.project_name}_output.csv')
        # if "regression" in dataset.challenge:
        #     path = os.path.join(
        #         output, f'{self.project_name}_regression_output.csv')
        # elif "classification" in dataset.challenge:
        #     path = os.path.join(
        #         output, f'{self.project_name}_classification_output.csv')

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
        test = self.test
        path = os.path.join(output, f'{self.project_name}_output.csv')
        # if "regression" in dataset.challenge:
        #     path = os.path.join(
        #         output, f'{self.project_name}_regression_output.csv')
        # elif "classification" in dataset.challenge:
        #     path = os.path.join(
        #         output, f'{self.project_name}_classification_output.csv')

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
        

    # def save_output(self, dataset, output='../leaderboard/'):
    #     # regression
    #     re = self.regression_leaderboard
    #     path_re = pathlib.Path(output) / f'{self.project_name}_regression_output.csv'
    #     header_re = ["challenge", "process_approach", "imputation", "patient_group", "drug_group", "train_test_rate", "remove_low_DAS", "random_state",
    #                  "model_id", "MAE", "MSE", "RMSE", 'R2', "Pearson_Correlation"]

    #     with open(path_re, 'a') as f:
    #         # create the csv writer
    #         writer = csv.writer(f)
    #         # write the header
    #         writer.writerow(header_re)
    #         for index, rows in re.iterrows():
    #             data = [dataset.challenge, dataset.process_approach, dataset.imputation, dataset.patient_group, dataset.drug_group, dataset.train_test_rate, dataset.random_state,
    #                     rows['model'], rows['MAE'], rows['MSE'], rows['RMSE'], rows['R2'], rows['Pearson_Correlation']]
    #             # write the data
    #             writer.writerow(data)

    #     # classification
    #     cl = self.regression_leaderboard
    #     path_cl = pathlib.Path(output) / \
    #         f'{self.project_name}_classification_output.csv'
    #     header_cl = ["challenge", "process_approach", "imputation", "patient_group", "drug_group", "train_test_rate", "remove_low_DAS", "random_state",
    #                  "model_id", "Accuracy", "F1-Score"]

    #     with open(path_cl, 'a') as f:
    #         # create the csv writer
    #         writer = csv.writer(f)
    #         # write the header
    #         writer.writerow(header_re)
    #         for index, rows in cl.iterrows():
    #             data = [dataset.challenge, dataset.process_approach, dataset.imputation, dataset.patient_group, dataset.drug_group, dataset.train_test_rate, dataset.random_state,
    #                     rows['model'], rows['MAE'], rows['MSE'], rows['RMSE'], rows['R2'], rows['Pearson_Correlation']]
    #             # write the data
    #             writer.writerow(data)
