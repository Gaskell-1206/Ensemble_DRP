import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import torch
from fancyimpute import (KNN, BiScaler, IterativeImputer, IterativeSVD,
                         NuclearNormMinimization, SimpleFill, SoftImpute)
from sklearn import preprocessing
pd.options.mode.chained_assignment = None

def calculate_DAS28_CRP(row):
    DAS28_CRP = 0.56*math.sqrt(row['tender_jts_28']) + 0.28*math.sqrt(
        row['swollen_jts_28']) + 0.014*row['pt_global_assess'] + 0.36*np.log(row['usresultsCRP']+1) + 0.96
    return DAS28_CRP

def responseClassify(row,baseline='DAS28_CRP_0M',next='DAS28_CRP_3M'):
    # set threshold
    lower_change = 0.6
    upper_change = 1.2
    
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


class CoronnaCERTAINDataset(torch.utils.data.Dataset):
    """CORRONA CERTAIN dataset."""

    def __init__(self,
                 library_root: Union[str, Path, os.PathLike],
                 challenge: Optional[Callable] = 'regression',
                 dataset: Optional[Callable] = 'CORRONA CERTAIN',
                 process_approach: Optional[Callable] = 'KVB',
                 imputation: Optional[Callable] = 'KNN',
                 patient_group: Optional[Callable] = ['bionaive TNF'],
                 drug_group: Optional[Callable] = 'all',
                 time_points: Optional[Callable] = (0, 3),
                 train_test_rate: float = 0.8,
                 remove_low_DAS = False,
                 save_csv: bool = False,
                 random_state: Optional[Callable] = 2022,
                 verbose: int = 0,
                 ):
        """Initializes instance of class CoronnaCERTAINDataset.

        Args:
            library_root: Path to the root director of library file
            challenge: regression (numerical, DAS-28CRP value),
                       regression_delta (numerical, changes of DAS-28CRP)
                       classification (categorical: Good, Moderate, No Response)
                       binary_classification (categorical: Responder, Non-responders)
            dataset: Dataset used for modeling ("CORRONA CERTAIN")
            process_approach: Dataset process approach. (KVB - from previous student ) (SC)
            imputation: imputation methods
            patient_group: Patient group
            drug_group: Medication group
            train_test_rate: sample rate for train and test set
            save_csv: If True, dataframe will be saved in "../Dataset/Coronna_Data_CERTAIN_{approach}_{time_1}M_{time_2}M_{subset}_{dataset}.csv"
        """
        if challenge not in ("regression", "regression_delta", "regression_delta_binary", "classification", "binary_classification"):
            raise ValueError(
                'challenge should be either "regression", "regression_delta", "regression_delta_binary", "classification", or "binary_classification"')
        if process_approach not in ("KVB", "SC"):
            raise ValueError('process_approach should be either "KVB" or "SC"')
        for patient_group_ in patient_group:
            if patient_group_ not in ("all", "bioexp nTNF", "bionaive TNF", "bionaive orencia", "KVB"):
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
        
        if challenge == "regression":
            self.target = "DAS28_CRP_3M"
        elif challenge == "regression_delta" or challenge == "regression_delta_binary":
            self.target = "delta"
        elif challenge == "classification":
            self.target = "DrugResponse"
        elif challenge == "binary_classification":
            self.target = "DrugResponse_binary"
        else:
            self.target = ""

        self.categorical = ["grp", "init_group", "gender", "final_education",
                            "race_grp", "ethnicity", "newsmoker", "drinker", "ara_func_class"]
        self.challenge = challenge
        self.process_approach = process_approach
        self.imputation = imputation
        self.patient_group = patient_group
        self.drug_group = drug_group
        self.time_points = time_points
        self.train_test_rate = train_test_rate
        self.remove_low_DAS = remove_low_DAS
        self.save_csv = save_csv
        self.sample_list = []
        self.random_state = random_state
        self.train_csv_loc = ''
        self.test_csv_loc = ''
        self.verbose = verbose

        if self.process_approach == 'KVB':
            df = pd.read_csv(self.library_root / "Analytical_Base_Table.csv")
            df = df.rename(columns={'3MResponse':'DrugResponse','DAS28-CRP_BL':'DAS28_CRP_0M','DAS28-CRP 3m':'DAS28_CRP_3M'})
            if self.challenge == "regression":
                df.insert(len(df.columns)-1,'DAS28_CRP_3M',df.pop('DAS28_CRP_3M'))
                df = df.drop(columns=['DrugResponse'])
            elif self.challenge == "regression_delta" or self.challenge == "regression_delta_binary":
                df.loc[:, 'delta'] = df['DAS28_CRP_0M'] - df['DAS28_CRP_3M']
                df = df.drop(columns=['DrugResponse','DAS28_CRP_3M'])
            elif self.challenge == 'classification':
                df.insert(len(df.columns)-1,'DrugResponse',df.pop('DrugResponse'))
                df = df.drop(columns=['DAS28_CRP_3M'])
            elif self.challenge == "binary_classification":
                df.loc[df['DrugResponse'] == 1, 'DrugResponse_binary'] = 1
                df.loc[df['DrugResponse'] == 0, 'DrugResponse_binary'] = 1
                df.loc[df['DrugResponse'] == 2, 'DrugResponse_binary'] = 0
                df = df.drop(columns=["DAS28_CRP_3M", "DrugResponse"])
                if self.verbose > 0:
                    print(f"Label Encoder, 0: Non-responders (No Response), 1: Responders(Good, Moderate)")
            self.df_train = df.sample(frac=self.train_test_rate, random_state=self.random_state)
            self.df_test = df.drop(self.df_train.index)

        elif self.process_approach == 'SC':
            if patient_group == 'KVB':
                self.sample_list = df_3M['UNMC_id']

            # drop columns
            df = df_all.drop(columns=['Unnamed: 62', 'Unnamed: 63'])
            
            # impute missing time-series rows
            df = self.transform_time_series_data(df)

            # train test split (only split rows without NaN in features used to calculate DAS-28CRP into testset)
            self.train_idx, self.test_idx, df = self.split_data(df)
            
            # feature engineering
            df = self.feature_engineering(df)
            
            # Imputation
            if self.imputation == None:
                # return raw
                imputed_train = df[df['UNMC_id'].isin(self.train_idx)]
                imputed_test = df[df['UNMC_id'].isin(self.test_idx)]
                
            else:
                imputed_train, imputed_test = self.Apply_imputation(df)

            # create dataframe by two consecutive months
            df_train = self.create_dataframe(imputed_train, 'Train')
            df_test = self.create_dataframe(imputed_test, 'Test')
                
            self.save_to_csv(df_train, "Train")
            self.save_to_csv(df_test, "Test")
            
            if self.save_csv:
                file_loc = os.path.join(
                    self.library_root, 'tableau_data', f'Coronna_Data_CERTAIN_{self.challenge}_{self.process_approach}_{self.time_points[0]}M_{self.time_points[1]}M_{self.patient_group}_{self.drug_group}', f'{dataset}.csv')
                file_loc = file_loc.replace(' ', '_') # avoid spacing
                self.check_dir(file_loc)
                if self.verbose > 0:
                    print("save file to:", file_loc)
                df_train_save = df_train
                df_train_save['dataset'] = 'Train'
                df_test_save = df_test
                df_test_save['dataset'] = 'Test'
                df = pd.concat([df_train_save, df_test_save])
                df.to_csv(file_loc, index=False)
                
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
        # data filter
        if (self.patient_group != 'all') & (self.patient_group != 'KVB'):
            df = df[df['init_group'].isin(self.patient_group)]
        if len(self.sample_list) > 0:
            if self.verbose > 0:
                print(self.sample_list)
            df = df[df['UNMC_id'].isin(self.sample_list)]
            
        # drop rows that have more than 10 NaN columns
        df = df.dropna(thresh=len(df.columns)-10, axis=0)

        df1 = df[df['futime'] == self.time_points[0]].drop(columns=['CDate'])
        df2 = df[df['futime'] == self.time_points[1]].drop(columns=['CDate'])
        
        df1_test = df1.dropna(axis=0, subset=['tender_jts_28', 'swollen_jts_28', 'pt_global_assess', 'usresultsCRP'])
        df2_test = df2.dropna(axis=0, subset=['tender_jts_28', 'swollen_jts_28', 'pt_global_assess', 'usresultsCRP'])
        # only put rows without NaN values in features for DAS-28 (3M) into test set
        overlap_index = df1_test.index.intersection(df2_test.index-1)
        test_n = int((1-self.train_test_rate) * len(df1))
        df1_test = df1_test.loc[overlap_index].sample(test_n, random_state=self.random_state)
        trainset = df1[~df1.index.isin(df1_test.index)]['UNMC_id']
        testset = df1[df1.index.isin(df1_test.index)]['UNMC_id']
        
        return trainset, testset, df
    
    def feature_engineering(self, df):
        # drop columns
        columns_drop = df.isnull().mean()[df.isnull().mean() > 0.7].index
        if self.verbose > 0:
            print("feature engineering, drop columns due to 70% missing value:",columns_drop)
        df = df.drop(columns=list(columns_drop))
        df = df.drop(columns=['CDate']) # useless
        
        # labelEncoder, tranform categorical features
        encoders = dict()
        for col_name in self.categorical:
            series = df[col_name].astype('str')
            label_encoder = preprocessing.LabelEncoder()
            df[col_name] = pd.Series(label_encoder.fit_transform(
                series[series.notnull()]), index=series[series.notnull()].index)
            df[col_name] = df[col_name].astype(int)
            encoders[col_name] = label_encoder
        
        return df

    def Apply_imputation(self,df):
        # use md to impute pt NaN
        # df['pt_global_assess'] = df.apply(lambda row: self.impute_pt_global_assess(row), axis=1)

        # filter train and test
        train_UNMC_id = df[df['UNMC_id'].isin(self.train_idx)]['UNMC_id']
        test_UNMC_id = df[df['UNMC_id'].isin(self.test_idx)]['UNMC_id']
        train = df[df['UNMC_id'].isin(self.train_idx)].drop(columns=['UNMC_id'])
        test = df[df['UNMC_id'].isin(self.test_idx)].drop(columns=['UNMC_id'])
        
        # imputation
        if self.imputation == 'SimpleFill':
            imputer = SimpleFill()
        elif self.imputation == 'KNN':
            imputer = KNN(k=15,verbose=False)
        elif self.imputation == 'SoftImpute':
            imputer = SoftImpute(verbose=False)
        elif self.imputation == 'NuclearNormMinimization':
            imputer = NuclearNormMinimization()
        elif self.imputation == 'BiScaler':
            imputer = BiScaler(verbose=False)
        elif self.imputation == 'IterativeImputer':
            imputer = IterativeImputer(verbose=False)
        elif self.imputation == 'IterativeSVD':
            imputer = IterativeSVD(verbose=False)

        # impute train set
        if self.verbose > 0:
            print("Missing values in train before imputation:", len(train[train.isna().any(axis=1)]))
        imputed_train = imputer.fit_transform(train)
        imputed_train_df = pd.DataFrame(imputed_train, columns=train.columns)
        imputed_train_df['UNMC_id'] = train_UNMC_id.values
        if self.verbose > 0:
            print("Missing values in train after imputation:", len(imputed_train_df[imputed_train_df.isna().any(axis=1)]))
        
        # impute test set
        if self.verbose > 0:
            print("Missing values in test before imputation:", len(test[test.isna().any(axis=1)]))
        imputed_test = imputer.transform(test)
        imput_test_df = pd.DataFrame(imputed_test, columns=test.columns)
        imput_test_df['UNMC_id'] = test_UNMC_id.values
        if self.verbose > 0:
            print("Missing values in test after imputation:", len(imput_test_df[imput_test_df.isna().any(axis=1)]))
        
        for col_name in self.categorical:
            imputed_train_df[col_name] = imputed_train_df[col_name].astype(int)
            imput_test_df[col_name] = imput_test_df[col_name].astype(int)
        
        return imputed_train_df, imput_test_df

    def check_dir(self, file_name):
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_dataframe(self, data_file, dataset):
        time_1, time_2 = self.time_points
        df1 = data_file[data_file['futime'] == time_1]
        df2 = data_file[data_file['futime'] == time_2]

        df1.loc[:, 'DAS28_CRP'] = df1.apply(lambda row: calculate_DAS28_CRP(row), axis=1)
        df2.loc[:, 'DAS28_CRP'] = df2.apply(lambda row: calculate_DAS28_CRP(row), axis=1)
        df2 = df2[['UNMC_id', 'DAS28_CRP']]
        
        # remove baseline DAS28 < 3.2
        if self.remove_low_DAS:
            df1 = df1[df1['DAS28_CRP'] > 3.2]
        
        # merge target
        df_merged = pd.merge(df1, df2, how="left", on="UNMC_id", suffixes=("_0M", "_3M"))
        # drop samples that can't be imputed
        df_merged = df_merged.dropna(axis=0, subset=['DAS28_CRP_3M'])

        # default regression tasks
        if self.challenge == "regression":
            pass
        # create delta for regression tasks
        elif self.challenge == "regression_delta" or self.challenge == "regression_delta_binary":
            df_merged.loc[:, 'delta'] = df_merged['DAS28_CRP_0M'] - df_merged['DAS28_CRP_3M']
            df_merged = df_merged.drop(columns="DAS28_CRP_3M")
        
        # create DrugResponse for 3-class classification tasks
        elif self.challenge == "classification":
            # df_merged.loc[:, 'delta'] = df_merged['DAS28_CRP_0M'] - df_merged['DAS28_CRP_3M']
            df_merged.loc[:, 'DrugResponse'] = df_merged.apply(lambda row: responseClassify(row), axis=1)
            df_merged = df_merged.drop(columns="DAS28_CRP_3M")
            # label encoder
            le = preprocessing.LabelEncoder()
            df_merged['DrugResponse'] = le.fit_transform(df_merged['DrugResponse'])
            inverse = le.inverse_transform([0,1,2])
            # print encoder mapping
            if self.verbose > 0:
                print(f"Label Encoder, 0:{inverse[0]}, 1:{inverse[1]}, 2:{inverse[0]}")
            
        # create DrugResponse_binary for binary classficaition tasks
        elif self.challenge == "binary_classification":
            df_merged.loc[:, 'DrugResponse'] = df_merged.apply(lambda row: responseClassify(row), axis=1)
            df_merged.loc[df_merged['DrugResponse'] == 'Good', 'DrugResponse_binary'] = 1
            df_merged.loc[df_merged['DrugResponse'] == 'Moderate', 'DrugResponse_binary'] = 1
            df_merged.loc[df_merged['DrugResponse'] == 'No Response', 'DrugResponse_binary'] = 0
            df_merged = df_merged.drop(columns=["DAS28_CRP_3M","DrugResponse"])
            if self.verbose > 0:
                print(f"Label Encoder, 0: Non-responders (No Response), 1: Responders(Good, Moderate)")

        # if self.patient_group != 'all':
        #     df_merged = df_merged.drop(columns='init_group')
        # subset
        subset = 'All'
        # if self.patient_group != 'all':
        #     df_merged = df_merged[df_merged['init_group']==self.patient_group]

        if len(self.sample_list) > 0:
            subset = 'KVB'
            df_merged = df_merged[df_merged['UNMC_id'].isin(self.sample_list)]

        df_merged = df_merged.drop(columns=['futime','UNMC_id'])

        return df_merged
    
    def save_to_csv(self, df, dataset):
        file_loc = os.path.join(
            self.library_root, '.csv_temp', f'{dataset}.csv')
        self.check_dir(file_loc)
        if self.verbose > 0:
            print("save file to:", file_loc)
        df.to_csv(file_loc, index=False)
            
        if dataset == 'Train':
            self.train_csv_loc = Path(file_loc)
        elif dataset == 'Test':
            self.test_csv_loc = Path(file_loc)

    def __len__(self):
        return len(self.df_out)

    def __getitem__(self, i: int):
        return self.df.iloc[i]

    def get_train(self):
        return self.df_train, self.train_csv_loc

    def get_test(self):
        return self.df_test, self.test_csv_loc

# dataset = CoronnaCERTAINDataset(
#     library_root='/Users/gaskell/Dropbox/Mac/Desktop/Autoimmune_Disease/Code/ML_RA_EHR/Dataset/',
#     challenge="regression",
#     dataset='CORRONA CERTAIN',
#     process_approach='SC',
#     imputation='KNN',
#     patient_group='bionaive TNF',
#     drug_group='all',
#     time_points=(0,3),
#     train_test_rate=0.8,
#     save_csv=False,
#     random_state=2022)

# train = dataset.get_train()
# test = dataset.get_test()
