import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import scale
pd.options.mode.chained_assignment = None

def calculate_DAS28_CRP(row):
    DAS28_CRP = 0.56*math.sqrt(row['tender_jts_28']) + 0.28*math.sqrt(
        row['swollen_jts_28']) + 0.014*row['pt_global_assess'] + 0.36*np.log(row['usresultsCRP']+1) + 0.96
    return DAS28_CRP

def responseClassify(row, baseline="DAS28_CRP_0M", next="DAS28_CRP_3M"):
    # set threshold
    low_change = 0.6
    high_change = 1.2

    change = row[baseline] - row[next]

    if change <= low_change:
        return "No Response"

    elif (change <= high_change) & (change > low_change):
        if row[next] > 5.1:
            return "No Response"
        else:
            return "Moderate"

    elif change > high_change:
        if row[next] > 3.2:
            return "Moderate"
        else:
            return "Good"

    else:
        return "Unknown"


class CoronnaCERTAINDataset(torch.utils.data.Dataset):
    """CORRONA CERTAIN dataset."""

    def __init__(self,
                 library_root: Union[str, Path, os.PathLike],
                 challenge: Optional[Callable] = 'regression',
                 dataset: Optional[Callable] = 'CORRONA CERTAIN',
                 process_approach: Optional[Callable] = 'KVB',
                 imputation: Optional[Callable] = 'KNN',
                 patient_group: Optional[Callable] = 'all',
                 drug_group: Optional[Callable] = 'all',
                 time_points: Optional[Callable] = (0, 3),
                 train_test_rate: float = 0.8,
                 save_csv: bool = False,
                 random_state: Optional[Callable] = 2022,
                 ):
        """Initializes instance of class CoronnaCERTAINDataset.

        Args:
            library_root: Path to the root director of library file
            challenge: regression(DAS-28CRP), classification(3MResponse), two_stage(both)
            dataset: Dataset used for modeling ("CORRONA CERTAIN")
            process_approach: Dataset process approach. (KVB - from previous student ) (SC)
            patient_group: Patient group
            drug_group: Medication group
            train_test_rate: sample rate for train and test set
            save_csv: If True, dataframe will be saved in "../Dataset/Coronna_Data_CERTAIN_{approach}_{time_1}M_{time_2}M_{subset}_{dataset}.csv"
        """
        if challenge not in ("regression", "classification", "two_stage"):
            raise ValueError(
                'challenge should be either "regression", "classification" or "two_stage"')
        if process_approach not in ("KVB", "SC"):
            raise ValueError('process_approach should be either "KVB" or "SC"')
        if patient_group not in ("all", "bioexp nTNF", "bionaive TNF", 'bionaive orencia', 'KVB'):
            raise ValueError(
                'patient_group should be either "all", "bioexp nTNF", "bionaive TNF", "bionaive orencia", "KVB"')
        if drug_group not in ("all", "actemra", "cimzia", "enbrel", "humira", "orencia", "remicade", "rituxan", "simponi"):
            raise ValueError(
                'drug_group should be "all", "actemra", "cimzia", "enbrel", "humira", "orencia", "remicade", "rituxan", "simponi"')

        self.library_root = Path(library_root)
        if dataset == 'CORRONA CERTAIN':
            library_name = 'Coronna_Data_CERTAIN_raw.csv'
            # check if file exists
            if not (self.library_root / library_name).is_file():
                excel_name = 'Coronna Data CERTAIN with KVB edits.xlsx'
                excel = pd.ExcelFile(self.library_root / excel_name)
                df_raw = pd.read_excel(excel, 'Sheet1')
                df_raw.to_csv(self.library_root / library_name, index=False)
            # read from csv will be much faster
            df_all = pd.read_csv(self.library_root / library_name)
            df_3M = pd.read_csv(self.library_root /
                                'Coronna_Data_CERTAIN_KVB_0M_3M.csv')

        self.categorical = ["grp", "init_group", "gender", "final_education",
                            "race_grp", "ethnicity", "newsmoker", "drinker", "ara_func_class"]
        self.challenge = challenge
        self.process_approach = process_approach
        self.imputation = imputation
        self.patient_group = patient_group
        self.drug_group = drug_group
        self.train_test_rate = train_test_rate
        self.save_csv = save_csv
        self.sample_list = []
        self.random_state = random_state
        self.train_csv_loc = ''
        self.test_csv_loc = ''

        if self.process_approach == 'KVB':
            if not (self.library_root / "Coronna_Data_CERTAIN_KVB_0M_3M.csv").is_file():
                excel_name = 'Coronna Data CERTAIN with KVB edits.xlsx'
                excel = pd.ExcelFile(self.library_root / excel_name)
                df_3M = pd.read_excel(excel, 'BL+3M')
                df_3M.to_csv(self.library_root /
                             "Coronna_Data_CERTAIN_KVB_0M_3M.csv", index=False)
            df_3M = pd.read_csv(self.library_root /
                                'Coronna_Data_CERTAIN_KVB_0M_3M.csv')
            # Need to be added for feature engineer!
            # df_3M = df_3M.drop(columns=[''])
            self.df_train = df_3M.sample(
                frac=self.train_test_rate, random_state=self.random_state)
            self.df_test = df_3M[-df_3M['UNMC_id']
                                 .isin(self.df_train['UNMC_id'])]

        elif self.process_approach == 'SC':
            if patient_group == 'KVB':
                self.sample_list = df_3M['UNMC_id']

            # drop columns
            df = df_all.drop(columns=['Unnamed: 62', 'Unnamed: 63'])
            
            # impute missing time-series rows
            df = self.transform_time_series_data(df)
        
            # train test split (only split rows without NaN in features used to calculate DAS-28CRP into testset)
            train_idx, test_idx, df = self.split_data(df)
            # Imputation
            if self.imputation == "KNN":
                imputed_train, imputed_test = self.KNN_imputation(
                    df, train_idx, test_idx)
            else:
                # return raw
                imputed_train = df[df['UNMC_id'].isin(train_idx)]
                imputed_test = df[df['UNMC_id'].isin(test_idx)]
                
            # print(imputed_train['futime'].value_counts())

            # create dataframe by two consecutive months
            df_train = self.create_dataframe(imputed_train, time_points, 'Train')
            df_test = self.create_dataframe(imputed_test, time_points, 'Test')
            self.df_train = df_train
            self.df_test = df_test

    def transform_time_series_data(self, df):
        # non_time_varying features
        df_dev_demo = df.loc[:, :"hx_anycancer"]
        grp = df_dev_demo.set_index('futime').groupby('UNMC_id')
        groups = [g.reindex(range(0, 15, 3)).fillna(
            method='ffill').reset_index() for _, g in grp]
        demo_out = pd.concat(groups, ignore_index=True).reindex(
            df_dev_demo.columns, axis=1)

        # time_varying features
        df_dev_time_varying = df.loc[:, "seatedbp1":]
        df_dev_time_varying['UNMC_id'] = df['UNMC_id']
        df_dev_time_varying['futime'] = df['futime']
        grp = df_dev_time_varying.set_index('futime').groupby('UNMC_id')
        groups = [g.reindex(range(0, 15, 3)).reset_index() for _, g in grp]
        time_varying_out = pd.concat(groups, ignore_index=True).reindex(
            df_dev_time_varying.columns, axis=1)
        time_varying_out['UNMC_id'] = time_varying_out['UNMC_id'].fillna(
            method='ffill')
        time_varying_out = time_varying_out.drop(columns=['futime', 'UNMC_id'])
        out_final = pd.concat([demo_out, time_varying_out], axis=1)
        return out_final

    def impute_pt_global_assess(self, row):
        if pd.isna(row['pt_global_assess']) & pd.notna(row['md_global_assess']):
            return row['md_global_assess']
        else:
            return row['pt_global_assess']

    def split_data(self, df):
        if self.patient_group != 'all':
            df = df[df['init_group'] == self.patient_group]
        if len(self.sample_list) > 0:
            df = df[df['UNMC_id'].isin(self.sample_list)]
            
        # drop columns
        columns_drop = df.isnull().mean()[df.isnull().mean() > 0.7].index
        print("feature engineering, drop columns due to 70% missing value:",columns_drop)
        df = df.drop(columns=list(columns_drop))
        # drop rows that have more than 10 NaN columns
        df = df.dropna(thresh=len(df.columns)-5, axis=0)

        df1 = df[df['futime'] == 0].drop(columns=['CDate'])
        df2 = df[df['futime'] == 3].drop(columns=['CDate'])
        df1_test = df1.dropna(axis=0, subset=[
                              'tender_jts_28', 'swollen_jts_28', 'pt_global_assess', 'usresultsCRP'])
        df2_test = df2.dropna(axis=0, subset=[
                              'tender_jts_28', 'swollen_jts_28', 'pt_global_assess', 'usresultsCRP'])
        # only put rows without NaN values in features for DAS-28 (3M) into test set
        overlap_index = df1_test.index.intersection(df2_test.index-1)
        test_n = int((1-self.train_test_rate) * len(df1))
        df1_test = df1_test.loc[overlap_index].sample(
            test_n, random_state=self.random_state)
        trainset = df1[~df1.index.isin(df1_test.index)]['UNMC_id']
        testset = df1[df1.index.isin(df1_test.index)]['UNMC_id']

        return trainset, testset, df

    def KNN_imputation(self, df, train_idx, test_idx):
        
        # tranform categorical features
        encoders = dict()
        for col_name in self.categorical:
            series = df[col_name].astype('str')
            label_encoder = preprocessing.LabelEncoder()
            df[col_name] = pd.Series(label_encoder.fit_transform(
                series[series.notnull()]), index=series[series.notnull()].index)
            encoders[col_name] = label_encoder
        # filter train and test
        train_UNMC_id = df[df['UNMC_id'].isin(train_idx)]['UNMC_id']
        test_UNMC_id = df[df['UNMC_id'].isin(test_idx)]['UNMC_id']
        train = df[df['UNMC_id'].isin(train_idx)].drop(
            columns=['UNMC_id', 'CDate'])
        test = df[df['UNMC_id'].isin(test_idx)].drop(
            columns=['UNMC_id', 'CDate'])
        # KNN imputation
        imputer = KNNImputer(n_neighbors=20)
        fit_train = imputer.fit(train)
        # impute train set
        imputed_train = fit_train.transform(train)
        imputed_train_df = pd.DataFrame(imputed_train, columns=train.columns)
        imputed_train_df['UNMC_id'] = train_UNMC_id.values
        # print("NaN in train:",imputed_train_df[imputed_train_df.isna().any(axis=1)])
        # impute test set
        imputed_test = fit_train.transform(test)
        imput_test_df = pd.DataFrame(imputed_test, columns=test.columns)
        imput_test_df['UNMC_id'] = test_UNMC_id.values

        return imputed_train_df, imput_test_df

    # def impute_DAS_28_features(self, df_to_impute):
    #     # impute pt_global_assess
    #     df_to_impute['pt_global_assess'] = df_to_impute.apply(
    #         lambda row: self.impute_pt_global_assess(row), axis=1)
    #     # drop NaN in feautres used for calculating DAS-28
    #     df_to_impute = df_to_impute.dropna(axis=0, subset=[
    #                                        'tender_jts_28', 'swollen_jts_28', 'pt_global_assess', 'usresultsCRP'], thresh=3)
    #     # impute 'tender_jts_28','swollen_jts_28','usresultsCRP' if only one missing
    #     df_for_imputation = df_to_impute.drop(columns=['UNMC_id'])

    #     # tranform categorical features
    #     encoders = dict()
    #     for col_name in self.categorical:
    #         series = df_for_imputation[col_name]
    #         label_encoder = preprocessing.LabelEncoder()
    #         df_for_imputation[col_name] = pd.Series(label_encoder.fit_transform(
    #             series[series.notnull()]), index=series[series.notnull()].index)
    #         encoders[col_name] = label_encoder
    #     df_for_imputation = df_for_imputation.reset_index(
    #         drop=True, inplace=True)

    #     imputer = KNNImputer(n_neighbors=50, weights="uniform")
    #     df_temp = pd.DataFrame(imputer.fit_transform(df_for_imputation))
    #     df_temp.columns = df_for_imputation.columns
    #     df_to_impute['tender_jts_28'] = df_temp['tender_jts_28']
    #     df_to_impute['swollen_jts_28'] = df_temp['swollen_jts_28']
    #     df_to_impute['pt_global_assess'] = df_temp['pt_global_assess']
    #     df_to_impute['usresultsCRP'] = df_temp['usresultsCRP']
    #     df_to_impute['DAS28_CRP'] = df_to_impute.apply(
    #         lambda row: calculate_DAS28_CRP(row), axis=1)

    #     return df_to_impute

    def check_dir(self, file_name):
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_dataframe(self, data_file, time_points, dataset):
        approach = 'SC'
        time_1, time_2 = time_points
        df1 = data_file[data_file['futime'] == time_1]
        df2 = data_file[data_file['futime'] == time_2]

        # imputed_1 = self.impute_DAS_28_features(df1)
        # imputed_2 = self.impute_DAS_28_features(df2)
        df1.loc[:, 'DAS28_CRP'] = df1.apply(
            lambda row: calculate_DAS28_CRP(row), axis=1).copy()
        df2.loc[:, 'DAS28_CRP'] = df2.apply(
            lambda row: calculate_DAS28_CRP(row), axis=1).copy()
        df2 = df2[['UNMC_id', 'DAS28_CRP']]
        # merge target
        df_merged = pd.merge(df1, df2, how="left",
                             on="UNMC_id", suffixes=("_0M", "_3M"))
        # drop samples that can't be imputed
        df_merged = df_merged.dropna(axis=0, subset=['DAS28_CRP_3M'])
        
        # print(df_merged[df_merged.isna().any(axis=1)])

        # create delta for regression tasks
        if self.challenge == "regression":
            pass
            # df_merged.loc[:, 'delta'] = df_merged['DAS28_CRP_3M'] - \
            #     df_merged['DAS28_CRP_0M']
        # create 3MResponse for classification tasks
        elif self.challenge == "classification":
            df_merged.loc[:, '3MResponse'] = df_merged.apply(
                lambda row: responseClassify(row), axis=1)
        # create two_stage data
        elif self.challenge == "two_stage":
            df_merged.loc[:, 'delta'] = df_merged['DAS28_CRP_3M'] - \
                df_merged['DAS28_CRP_0M']
            df_merged.loc[:, '3MResponse'] = df_merged.apply(
                lambda row: responseClassify(row), axis=1)
        # df_merged = df_merged.drop(columns="DAS28_CRP_3M")

        if self.patient_group != 'all':
            df_merged = df_merged.drop(columns='init_group')
        # subset
        subset = 'All'
        # if self.patient_group != 'all':
        #     df_merged = df_merged[df_merged['init_group']==self.patient_group]

        if len(self.sample_list) > 0:
            subset = 'KVB'
            df_merged = df_merged[df_merged['UNMC_id'].isin(self.sample_list)]

        df_merged = df_merged.drop(columns=['futime','UNMC_id'])
        # df_merged = df_merged.drop(columns=['futime'])

        if self.save_csv:
            file_loc = os.path.join(
                self.library_root, f'Coronna_Data_CERTAIN_{self.challenge}_{self.process_approach}_{time_1}M_{time_2}M_{self.patient_group}_{self.drug_group}', f'{dataset}.csv')
            file_loc.replace(' ', '_') # avoid spacing
            self.check_dir(file_loc)
            print("save file to:", file_loc)
            df_merged.to_csv(file_loc, index=False)
        else:
            file_loc = os.path.join(
                self.library_root, '.csv_temp', f'{dataset}.csv')
            self.check_dir(file_loc)
            print("save file to:", file_loc)
            df_merged.to_csv(file_loc, index=False)
            
        if dataset == 'Train':
            self.train_csv_loc = Path(file_loc)
        elif dataset == 'Test':
            self.test_csv_loc = Path(file_loc)

        return df_merged

    def __len__(self):
        return len(self.df_out)

    def __getitem__(self, i: int):
        return self.df.iloc[i]

    def get_train(self):
        return self.df_train, self.train_csv_loc

    def get_test(self):
        return self.df_test, self.test_csv_loc

# dataset = CoronnaCERTAINDataset(
#     library_root = '/Users/gaskell/Dropbox/Mac/Desktop/Autoimmune_Disease/Code/ML_RA_EHR/Dataset/',
#     challenge = "classification",
#     dataset = 'CORRONA CERTAIN',
#     process_approach = 'SC',
#     patient_group = 'bionaive TNF',
#     drug_group = 'all',
#     save_csv = True,
#     )

# train = dataset.get_train()
# test = dataset.get_test()
