import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.infogram import H2OInfogram
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from DataModule.Data_Preparation import CoronnaCERTAINDataset
pd.options.mode.chained_assignment = None

if __name__ == "__main__":
    # define data module
    dataset = CoronnaCERTAINDataset(
        library_root='/Users/gaskell/Dropbox/Mac/Desktop/Autoimmune_Disease/Code/ML_RA_EHR/Dataset/',
        challenge="classification",
        dataset='CORRONA CERTAIN',
        process_approach='SC',
        patient_group='bionaive TNF',
        drug_group='all',
        save_csv=True
    )

    train = dataset.get_train()
    # Start the H2O cluster (locally)
    h2o.init()

    # Import a sample binary outcome train/test set into H2O
    train = h2o.upload_file("./Dataset/.csv_temp/Train.csv")
    test = h2o.upload_file("./Dataset/.csv_temp/Test.csv")

    # Identify predictors and response
    x = train.columns[:-1]
    y = "3MResponse"

    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    # Run AutoML for 20 base models
    aml = H2OAutoML(max_models=20, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    # View the AutoML Leaderboard
    lb = aml.leaderboard
    lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
