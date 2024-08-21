import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, sep=",")
    convert_col = ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    for col in convert_col:
        df[col] = df[col].astype('category')
    return df

class DataSplitter:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.df, test_size=test_size, random_state=random_state, stratify=self.df[self.target])

class Preprocessor:
    def __init__(self, trainset, testset, target):
        self.trainset = trainset
        self.testset = testset
        self.target = target
        self.preprocessor = None

    def split_features_target(self):
        X_train = self.trainset.drop(columns=[self.target])
        y_train = self.trainset[self.target]
        X_test = self.testset.drop(columns=[self.target])
        y_test = self.testset[self.target]
        return X_train, X_test, y_train, y_test

    def preprocess(self, numeric_features, categorical_features):
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        numeric_transformer = Pipeline(steps=[('MinMax', MinMaxScaler()), ('quantil', QuantileTransformer())])

        self.preprocessor = ColumnTransformer(transformers=[
            ('MinMax_Q', numeric_transformer, numeric_features),
            ('OHE', cat_transformer, categorical_features)
        ], remainder='passthrough')

        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor)])
        X_train = pipeline.fit_transform(self.trainset.drop(columns=[self.target]))
        X_test = pipeline.transform(self.testset.drop(columns=[self.target]))
        return X_train, X_test

    def get_feature_names(self):
        if self.preprocessor is None:
            raise ValueError("Le préprocesseur n'a pas été initialisé. Assurez-vous d'avoir appelé la méthode preprocess avant d'accéder aux noms des caractéristiques.")
        return self.preprocessor.get_feature_names_out()

def preprocess_data(file_path="BankChurners.csv"):
    df = load_data(file_path)
    target = 'Attrition_Flag'
    splitter = DataSplitter(df, target)
    trainset, testset = splitter.split_data()
    preprocessor = Preprocessor(trainset, testset, target)
    X_train, X_test, y_train, y_test = preprocessor.split_features_target()

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
    X_train_trans, X_test_trans = preprocessor.preprocess(numeric_features, categorical_features)
    feature_names = preprocessor.get_feature_names()

    y_train = y_train.map({'Attrited Customer': 1, 'Existing Customer': 0})
    y_test = y_test.map({'Attrited Customer': 1, 'Existing Customer': 0})

    X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names).drop(["MinMax_Q__CLIENTNUM"], axis=1)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names).drop(["MinMax_Q__CLIENTNUM"], axis=1)

    cols = ["MinMax_Q__Contacts_Count_12_mon", 
            "MinMax_Q__Total_Relationship_Count", 
            "MinMax_Q__Total_Amt_Chng_Q4_Q1", 
            "MinMax_Q__Total_Ct_Chng_Q4_Q1", 
            "MinMax_Q__Total_Revolving_Bal", 
            "MinMax_Q__Total_Trans_Ct", 
            "MinMax_Q__Total_Trans_Amt", 
            "MinMax_Q__Months_Inactive_12_mon", 
            "MinMax_Q__Customer_Age"]

    X_train_trans = X_train_trans_df[cols].to_numpy()
    X_test_trans = X_test_trans_df[cols].to_numpy()

    y_test = y_test.reset_index(drop=True)
    X_test_trans_df = X_test_trans_df.reset_index(drop=True)
    testset = testset.reset_index(drop=True)

    return X_train_trans, X_test_trans, y_train, y_test, cols, X_test_trans_df, testset

if __name__ == "__main__":
    X_train_trans, X_test_trans, y_train, y_test, feature_names, X_test_trans_df, testset = preprocess_data()
    print(X_train_trans.shape, X_test_trans.shape, y_train.shape, y_test.shape)
    print("Feature names:", feature_names)
