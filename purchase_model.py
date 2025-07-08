import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import SMOTE

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df['Weekend'] = df['Weekend'].replace({True: 1, False: 0})
    df['Revenue'] = df['Revenue'].replace({True: 1, False: 0})
    df['Returning_Visitor'] = np.where(df['VisitorType'] == 'Returning_Visitor', 1, 0)
    df.drop(columns=['VisitorType'], inplace=True)
    if df['Month'].dtype == 'object':
        df['Month'] = OrdinalEncoder().fit_transform(df[['Month']])
    return df

def build_pipeline(X, model=None):
    num_cols = X.select_dtypes(exclude='object').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant')),
        ('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, num_cols),
        ('categorical', categorical_pipeline, cat_cols)
    ], remainder='passthrough')

    final_model = model if model else MLPClassifier(hidden_layer_sizes=(27, 50), max_iter=300, activation='relu', solver='adam', random_state=1)

    pipeline = IMBPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=1)),
        ('feature_selection', SelectKBest(score_func=chi2, k=6)),
        ('model', final_model)
    ])
    return pipeline

def train_and_evaluate(df):
    X = df.drop(columns=['Revenue'])
    y = df['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }
    return pipeline, metrics, X.columns.tolist()
