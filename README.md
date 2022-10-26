# Ensemble DRP

Ensmeble DRP is Ensemble Learning for Drug Response Prediction with its primary aim to use ensemble machine learning methods to predict patients's response to drugs using tabular electronic health records. Specifically, we built a pipeline for data preprocessing and two-stage approach for modeling drug effectiveness.

## Datasets

The study was conducted using tabular electronic health records (EHR) data gathered from the Comparative Effectiveness Registry to Study Therapies for Arthritis and Inflammatory Conditions ([CERTAIN](https://www.corevitas.com/)).
More information about CORRONA CERTAIN can be found in multiple publications:

+ CORRONA: [The CORRONA database](http://dx.doi.org/10.1136/ard.2005.043497)
+ CERTAIN: [Design characteristics of the CORRONA CERTAIN study: a comparative effectiveness study of biologic agents for rheumatoid arthritis patients](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3978136/)

+ The toy dataset is provided 

## Dependencies and Installation

First install H2O according to the directions at [H2O.ai Website](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html) for your operating system. 

### Installing Directly from Source

Clone the repository, navigate to Ensemble_DRP root directory and run

```bash
pip install -e .
```

All dependencies will be installed using packages in `requirements.txt`.

## Explore Files

+ [Data_Preparation.py](https://github.com/Gaskell-1206/Ensemble_DRP/blob/H2O-Incoporation/DataModule/Data_Preparation.py): data preprocessing pipeline, including feature engineering, missing value imputation, train test split.
+ [H2O_experiment.py](https://github.com/Gaskell-1206/Ensemble_DRP/blob/H2O-Incoporation/ModelModule/H2O_experiment.py): incorporated h2o auto machine learning framework, including base model families and stakced ensemble learning.
+ [Ensemble_DRP_visualization.ipynb](https://github.com/Gaskell-1206/Ensemble_DRP/blob/H2O-Incoporation/Ensemble_DRP_visualization.ipynb): model performance comparison and visualization.
+ [Coronna_CERTAIN_toy.csv](https://github.com/Gaskell-1206/Ensemble_DRP/blob/main/Dataset/Coronna_CERTAIN_toy.csv): toy dataset created by shuffle values of each column in the real dataset for privacy.
