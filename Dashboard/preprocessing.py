import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def load_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    Args:
        file_path (str): Chemin vers le fichier CSV.
    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.
    """
    df = pd.read_csv(file_path, sep=",")

    # Conversion de certaines colonnes en "category"
    convert_col = ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    for col in convert_col:
        df[col] = df[col].astype('category')

    return df

class DataSplitter:
    """
    Classe utilisée pour diviser les données en ensemble d'entraînement et de test.
    """
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.trainset = None
        self.testset = None

    def split_data(self, test_size=0.2, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test.
        Args:
            test_size (float): Proportion de l'ensemble de test.
            random_state (int): État aléatoire pour la reproductibilité.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Ensemble d'entraînement et ensemble de test.
        """
        self.trainset, self.testset = train_test_split(self.df, test_size=test_size, random_state=random_state, stratify=self.df[self.target])
        return self.trainset, self.testset

class Preprocessor:
    """
    Classe utilisée pour prétraiter les données et les diviser en X_train, X_test, y_train, et y_test.
    """
    def __init__(self, trainset, testset, target):
        self.trainset = trainset
        self.testset = testset
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def split_features_target(self):
        """
        Divise les données en caractéristiques (features) et cibles (target).
        Returns:
            Tuple: X_train, X_test, y_train, y_test.
        """
        self.X_train = self.trainset.drop(columns=[self.target])
        self.y_train = self.trainset[self.target]
        self.X_test = self.testset.drop(columns=[self.target])
        self.y_test = self.testset[self.target]
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self, numeric_features, categorical_features):
        """
        Applique le prétraitement aux données.
        Args:
            numeric_features (list): Liste des caractéristiques numériques.
            categorical_features (list): Liste des caractéristiques catégorielles.
        Returns:
            Tuple: X_train prétraité, X_test prétraité.
        """
        cat_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        numeric_transformer = Pipeline(steps=[
            ('MinMax', MinMaxScaler()),
            ('quantil', QuantileTransformer())
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('MinMax_Q', numeric_transformer, numeric_features),
                ('OHE', cat_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor)])
        self.X_train = pipeline.fit_transform(self.X_train)
        self.X_test = pipeline.transform(self.X_test)
        return self.X_train, self.X_test

    def get_feature_names(self):
        """
        Récupère les noms des caractéristiques après le prétraitement.
        Returns:
            np.array: Noms des caractéristiques.
        """
        if self.preprocessor is None:
            raise ValueError("Le préprocesseur n'a pas été appliqué. Veuillez exécuter la méthode preprocess d'abord.")

        feature_names = self.preprocessor.get_feature_names_out()
        return feature_names

def main():
    # Charger les données
    df = load_data("BankChurners.csv")

    # Diviser les données
    target = 'Attrition_Flag'
    splitter = DataSplitter(df, target)
    trainset, testset = splitter.split_data()

    # Prétraiter les données
    preprocessor = Preprocessor(trainset, testset, target)
    X_train, X_test, y_train, y_test = preprocessor.split_features_target()

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()

    X_train_trans, X_test_trans = preprocessor.preprocess(numeric_features=numeric_features, categorical_features=categorical_features)

    # Récupérer les noms des caractéristiques après prétraitement
    feature_names = preprocessor.get_feature_names()

    # Convertir les cibles
    y_train = y_train.map({'Attrited Customer': 1, 'Existing Customer': 0})
    y_test = y_test.map({'Attrited Customer': 1, 'Existing Customer': 0})

    # Créer des DataFrames intermédiaires pour conserver les noms des colonnes associées
    X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names).drop(["MinMax_Q__CLIENTNUM"], axis=1)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names).drop(["MinMax_Q__CLIENTNUM"], axis=1)

    # Sélectionner les colonnes spécifiques pour la modélisation
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

    # Réinitialiser les indices
    y_test = y_test.reset_index(drop=True)
    X_test_trans_df = X_test_trans_df.reset_index(drop=True)
    testset = testset.reset_index(drop=True)

    return X_train_trans, X_test_trans, y_train, y_test, cols, X_test_trans_df, testset

if __name__ == "__main__":
    X_train_trans, X_test_trans, y_train, y_test, feature_names, X_test_trans_df, testset = main()
    print(X_train_trans.shape, X_test_trans.shape, y_train.shape, y_test.shape)
    print("Feature names:", feature_names)
