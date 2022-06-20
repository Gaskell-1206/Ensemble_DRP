import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import scale
pd.options.mode.chained_assignment = None

class CoronnaCERTAINDataset():
    """CORRONA CERTAIN dataset."""
    def __init__(self,
                 library_root: Union[str, Path, os.PathLike],
                 challenge: Optional[Callable] = 'DAS-28CRP',
                 dataset: Optional[Callable] = 'CORRONA CERTAIN',
                 process_approach: Optional[Callable] = 'KVB',
                 patient_group: Optional[Callable] = 'all',
                 drug_group: Optional[Callable] = 'all',
                 time_points: Optional[Callable] = (0,3),
                 train_test_rate: float = 0.8,
                 save_csv: bool = False,
                 random_state: Optional[Callable] = 2022,
                 ):
        """Initializes instance of class CoronnaCERTAINDataset.
        
        Args:
            library_root: Path to the root director of library file
            challenge: regression(DAS-28CRP), classification(3MResponse), two_stage(both)
            dataset: Dataset used for modeling ("CORRONA CERTAIN")
            process_approach: Dataset process approach. 
            patient_group: Patient group
            drug_group: Medication group
            train_test_rate: sample rate for train and test set
            save_csv: If True, dataframe will be saved in "../Dataset/Coronna_Data_CERTAIN_{approach}_{time_1}M_{time_2}M_{subset}_{dataset}.csv"
        """
        if challenge not in ("regression", "classification", "two_stage"):
            raise ValueError('challenge should be either "regression", "classification" or "two_stage"')
        if process_approach not in ("KVB", "SC"):
            raise ValueError('process_approach should be either "KVB" or "SC"')
        if patient_group not in ("all","bioexp nTNF", "bionaive TNF", 'bionaive orencia'):
            raise ValueError('patient_group should be either "all", "bioexp nTNF", "bionaive TNF" or "bionaive orencia"')
        if drug_group not in ("all","actemra", "cimzia", "enbrel", "humira", "orencia", "remicade", "rituxan", "simponi"):
            raise ValueError('drug_group should be "all", "actemra", "cimzia", "enbrel", "humira", "orencia", "remicade", "rituxan", "simponi"')
        
        self.library_root = Path(library_root)
        if dataset == 'CORRONA CERTAIN':
            # library_name = 'Coronna Data CERTAIN with KVB edits.xlsx'
            library_name = 'Coronna_Data_CERTAIN_raw.csv'
            # excel = pd.ExcelFile(self.library_root / library_name)
            # df_all = pd.read_excel(excel, 'Sheet1')
            # df_3M = pd.read_excel(excel, 'BL+3M')
            df_all = pd.read_csv(self.library_root / library_name)
            df_3M = pd.read_csv(self.library_root / 'Coronna_Data_CERTAIN_KVB_0M_3M.csv')

        self.categorical = ["grp", "init_group", "ara_func_class", "gender", "final_education", "race_grp", "newsmoker", "drinker"]
        self.challenge = challenge
        self.process_approach = process_approach
        self.patient_group = patient_group
        self.drug_group = drug_group
        self.train_test_rate = train_test_rate
        self.save_csv = save_csv
        self.sample_list = []
        self.random_state = random_state
        if process_approach == 'KVB':
            self.sample_list = df_3M['UNMC_id']
        
        # drop columns
        df = df_all.drop(columns=['Unnamed: 62', 'Unnamed: 63'])
        # impute missing time-series rows 
        df = self.transform_time_series_data(df)
        # train test split (only split rows without NaN in features used to calculate DAS-28CRP into testset)
        train_idx, test_idx = self.split_data(df)
        # Imputation
        imputed_train, imputed_test = self.KNN_imputation(df,train_idx,test_idx)
        # create dataframe by two consecutive months
        df_train = self.create_dataframe(imputed_train, time_points, 'Train')
        df_test = self.create_dataframe(imputed_test, time_points, 'Test')
        self.df_train = df_train
        self.df_test = df_test
        
    def transform_time_series_data(self, df):
        # non_time_varying features
        df_dev_demo = df.loc[:,:"hx_anycancer"]
        grp = df_dev_demo.set_index('futime').groupby('UNMC_id')
        groups = [g.reindex(range(0, 15, 3)).fillna(method='ffill').reset_index() for _, g in grp]
        demo_out = pd.concat(groups, ignore_index=True).reindex(df_dev_demo.columns, axis=1)
        
        # time_varying features
        df_dev_time_varying = df.loc[:,"seatedbp1":]
        df_dev_time_varying['UNMC_id'] = df['UNMC_id']
        df_dev_time_varying['futime'] = df['futime']
        grp = df_dev_time_varying.set_index('futime').groupby('UNMC_id')
        groups = [g.reindex(range(0, 15, 3)).reset_index() for _, g in grp]
        time_varying_out = pd.concat(groups, ignore_index=True).reindex(df_dev_time_varying.columns, axis=1)
        time_varying_out['UNMC_id'] = time_varying_out['UNMC_id'].fillna(method='ffill')
        time_varying_out = time_varying_out.drop(columns=['futime','UNMC_id'])
        out_final = pd.concat([demo_out,time_varying_out],axis=1)
        return out_final
        
    def calculate_DAS28_CRP(self, row):
        DAS28_CRP = 0.56*math.sqrt(row['tender_jts_28']) + 0.28*math.sqrt(row['swollen_jts_28']) + 0.014*row['pt_global_assess'] + 0.36*np.log(row['usresultsCRP']+1) + 0.96
        return DAS28_CRP
    
    def responseClassify(self, row):
        # set threshold
        low_change = 0.65
        high_change = 1.25
        
        change = row['DAS28_CRP_0M'] - row['DAS28_CRP_3M']

        if change <= low_change:
            return "No Response"
        
        elif (change <= high_change) & (change > low_change):
            if row['DAS28_CRP_3M'] > 5.1:
                return "No Response"
            else:
                return "Moderate"
        
        elif change > high_change:
            if row['DAS28_CRP_3M'] > 3.2:
                return "Moderate"
            else:
                return "Good"

        else:
            return "Unknown"
    
    def impute_pt_global_assess(self, row):
        if pd.isna(row['pt_global_assess']) & pd.notna(row['md_global_assess']):
            return row['md_global_assess']
        else:
            return row['pt_global_assess']
        
    def split_data(self, df):
        if self.patient_group != 'all':
            df = df[df['init_group']==self.patient_group]
        if len(self.sample_list)>0:
            df = df[df['UNMC_id'].isin(self.sample_list)]
            
        df1 = df[df['futime']==0].drop(columns=['CDate'])
        df2 = df[df['futime']==3].drop(columns=['CDate'])
        df1_test = df1.dropna(axis=0, subset=['tender_jts_28','swollen_jts_28','pt_global_assess','usresultsCRP'])
        df2_test = df2.dropna(axis=0, subset=['tender_jts_28','swollen_jts_28','pt_global_assess','usresultsCRP'])
        overlap_index = df1_test.index.intersection(df2_test.index-1)
        test_n = int((1-self.train_test_rate) * len(df1))
        df1_test = df1_test.loc[overlap_index].sample(test_n,random_state=self.random_state)
        trainset = df1[~df1.index.isin(df1_test.index)]['UNMC_id']
        testset = df1[df1.index.isin(df1_test.index)]['UNMC_id']
        
        return trainset, testset
        
    def KNN_imputation(self, df, train_idx, test_idx):
        # drop columns
        columns_drop = df.isnull().mean()[df.isnull().mean() > 0.7].index
        df = df.drop(columns=list(columns_drop))
        # tranform categorical features
        encoders = dict()
        for col_name in self.categorical:
            series = df[col_name]
            label_encoder = preprocessing.LabelEncoder()
            df[col_name] = pd.Series(label_encoder.fit_transform(series[series.notnull()]),index=series[series.notnull()].index)
            encoders[col_name] = label_encoder
        # filter train and test
        train_UNMC_id = df[df['UNMC_id'].isin(train_idx)]['UNMC_id']
        test_UNMC_id = df[df['UNMC_id'].isin(test_idx)]['UNMC_id']                
        train = df[df['UNMC_id'].isin(train_idx)].drop(columns=['UNMC_id','CDate'])
        test = df[df['UNMC_id'].isin(test_idx)].drop(columns=['UNMC_id','CDate'])
        # KNN imputation
        imputer = KNNImputer(n_neighbors=15)
        fit_train = imputer.fit(train)
        imputed_train = fit_train.transform(train)
        imputed_train_df = pd.DataFrame(imputed_train, columns = train.columns)
        imputed_test = fit_train.transform(test)                 
        imput_test_df = pd.DataFrame(imputed_test, columns = test.columns)
        imputed_train_df['UNMC_id'] = train_UNMC_id.values
        imput_test_df['UNMC_id'] = test_UNMC_id.values
        # imputed_train_df.info()
        # imput_test_df.info()
        return imputed_train_df, imput_test_df
    
    def impute_DAS_28_features(self, df_to_impute):
        # impute pt_global_assess
        df_to_impute['pt_global_assess'] = df_to_impute.apply(lambda row: self.impute_pt_global_assess(row), axis = 1)
        # drop NaN in feautres used for calculating DAS-28
        df_to_impute = df_to_impute.dropna(axis=0, subset=['tender_jts_28','swollen_jts_28','pt_global_assess','usresultsCRP'], thresh=3)
        # impute 'tender_jts_28','swollen_jts_28','usresultsCRP' if only one missing
        df_for_imputation = df_to_impute.drop(columns=['UNMC_id'])
        
        # tranform categorical features
        encoders = dict()
        for col_name in self.categorical:
            series = df_for_imputation[col_name]
            label_encoder = preprocessing.LabelEncoder()
            df_for_imputation[col_name] = pd.Series(label_encoder.fit_transform(series[series.notnull()]),index=series[series.notnull()].index)
            encoders[col_name] = label_encoder
        df_for_imputation = df_for_imputation.reset_index(drop=True,inplace=True)

        imputer = KNNImputer(n_neighbors=50, weights="uniform")
        df_temp = pd.DataFrame(imputer.fit_transform(df_for_imputation))
        df_temp.columns = df_for_imputation.columns
        df_to_impute['tender_jts_28'] = df_temp['tender_jts_28']
        df_to_impute['swollen_jts_28'] = df_temp['swollen_jts_28']
        df_to_impute['pt_global_assess'] = df_temp['pt_global_assess']
        df_to_impute['usresultsCRP'] = df_temp['usresultsCRP']
        df_to_impute['DAS28_CRP'] = df_to_impute.apply(lambda row: self.calculate_DAS28_CRP(row), axis = 1)

        return df_to_impute
    
    def create_dataframe(self, data_file, time_points, dataset):
        approach = 'SC'
        time_1, time_2 = time_points
        df1 = data_file[data_file['futime']==time_1]
        df2 = data_file[data_file['futime']==time_2]
        
        # imputed_1 = self.impute_DAS_28_features(df1)
        # imputed_2 = self.impute_DAS_28_features(df2)
        df1.loc[:,'DAS28_CRP'] = df1.apply(lambda row: self.calculate_DAS28_CRP(row), axis = 1).copy()
        df2.loc[:,'DAS28_CRP'] = df2.apply(lambda row: self.calculate_DAS28_CRP(row), axis = 1).copy()
        df2 = df2[['UNMC_id','DAS28_CRP']]
        # merge target
        df_merged = pd.merge(df1, df2, how="right", on="UNMC_id", suffixes= ("_0M","_3M"))
        
        # create delta for regression tasks
        if self.challenge == "regression":
            df_merged.loc[:,'delta'] = df_merged['DAS28_CRP_3M'] - df_merged['DAS28_CRP_0M']
            df_merged = df_merged.drop(columns="DAS28_CRP_3M")
        # create 3MResponse for classification tasks
        elif self.challenge == "classification":
            df_merged.loc[:,'3MResponse'] = df_merged.apply(lambda row: self.responseClassify(row), axis = 1)
            df_merged = df_merged.drop(columns="DAS28_CRP_3M")
        else:
            df_merged.loc[:,'delta'] = df_merged['DAS28_CRP_3M'] - df_merged['DAS28_CRP_0M']
            df_merged.loc[:,'3MResponse'] = df_merged.apply(lambda row: self.responseClassify(row), axis = 1)
            df_merged = df_merged.drop(columns="DAS28_CRP_3M")
            
        if self.patient_group != 'all':
            df_merged = df_merged.drop(columns='init_group')
        # subset
        subset = 'All'
        # if self.patient_group != 'all':
        #     df_merged = df_merged[df_merged['init_group']==self.patient_group]
        
        if len(self.sample_list)>0:
            subset = 'KVB'
            df_merged = df_merged[df_merged['UNMC_id'].isin(self.sample_list)]
            
        df_merged = df_merged.drop(columns=['futime','UNMC_id'])
        
        if self.save_csv:
            file_loc = os.path.join(self.library_root, f'Coronna_Data_CERTAIN_{self.challenge}_{self.process_approach}_{time_1}M_{time_2}M_{self.patient_group}_{self.drug_group}_{dataset}.csv')
            print("save file to:",file_loc)
            df_merged.to_csv(file_loc,index=False)
        return df_merged
    
    def __len__(self):
        return len(self.df_out)
    
    def __getitem__(self, i: int):
        return self.df.iloc[i]
    
    def get_df_train(self):
        return self.df_train
    
    def get_df_test(self):
        return self.df_test
    
# dataset = CoronnaCERTAINDataset(
#     library_root = '/Users/gaskell/Dropbox/Mac/Desktop/Autoimmune_Disease/Code/ML_RA_EHR/Dataset/',
#     challenge = "classification",
#     dataset = 'CORRONA CERTAIN',
#     process_approach = 'SC',
#     patient_group = 'bionaive TNF',
#     drug_group = 'all',
#     save_csv = True,
#     )

# train = dataset.get_df_train()
# test = dataset.get_df_test()
