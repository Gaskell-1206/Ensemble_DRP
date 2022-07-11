from pandas.api.types import is_list_like
class ensamble_model: 
    def __init__(self, model_list, ensamble_type, meta_learner):
        if is_list_like(model_list)==False:
            raise ValueError("model_list must be list like")
        for model in model_list:
            if model not in ('KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes"):
                raise ValueError("model must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes")
        if meta_learner not in ('KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes", None):
            raise ValueError("meta_learner must be ", 'KNN', "Bayes", 'Logistic', 'Linear', 'Ridge', 'Lasso', "SVM", "Bayes")
        if ensamble_type not in ('Stacking', 'Bagging', 'ADA-Boost', 'Grad-Boost'):
            raise ValueError("ensamble_type must be in ", 'Stacking', 'Bagging', 'ADA-Boost', 'Grad-Boost')
        if len(model_list)!=1 and ensamble_type in ( 'Bagging', 'ADA-Boost', 'Grad-Boost'):
            raise ValueError(ensamble_type, 'requires a homogeneous weak learner ')

