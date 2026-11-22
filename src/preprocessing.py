"""All learned imputation, scale and category maps fit inside training folds."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
def make_preprocessor(columns,categorical):
    cat=[c for c in columns if c in categorical]; num=[c for c in columns if c not in categorical]
    return ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),num),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],verbose_feature_names_out=False)
