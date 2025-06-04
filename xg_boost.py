import pandas as pd
import numpy as np
from numpy import nan

from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

import xgboost as xgb

def process(filepath):
    # 1. Load dataset
    data = pd.read_csv(filepath)
    result1 = data.copy()  # Original dataset
    data_before_cleaning = data.copy()  # For API return

    # 2. Replace 0s with NaNs for selected features
    data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, nan)
    df_pima = data

    # 3. Impute missing values
    imputer = IterativeImputer()
    df_pima[["Insulin", "SkinThickness", "Glucose", "BloodPressure", "BMI"]] = imputer.fit_transform(df_pima[["Insulin","SkinThickness", "Glucose", "BloodPressure", "BMI"]])
    

    #Outlier
    model1 = LocalOutlierFactor(n_neighbors=10)
    # model fitting
    y_pred = model1.fit_predict(df_pima)
    # filter outlier index
    not_outlier_index = np.where(y_pred == 1)
    outlier_index = np.where(y_pred == -1)# negative values are outliers and positives inliers
    # filter outlier values
    df_pima_1 = df_pima.iloc[not_outlier_index]


    X = df_pima_1.drop(columns=["Outcome"])
    y = df_pima_1['Outcome']

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size= 0.20, random_state=42, stratify= y)

    #Create a Gaussian Classifier
    clf=xgb.XGBClassifier()

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuraryXGBoost = metrics.accuracy_score(y_test, y_pred)

    #SMOTE X XG BOOST
    sm = SMOTE(random_state=0)
    X_SMOTE, y_SMOTE = sm.fit_resample(X_train, y_train.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, random_state=42, test_size= 0.20)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuraryXGBoostAndSMOTE = metrics.accuracy_score(y_test, y_pred)

    # SMOTETomek X XG BOOST

    sm = SMOTETomek(random_state=0)
    X_SMOTETOMEK, y_SMOTETOMEK = sm.fit_resample(X_train, y_train.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X_SMOTETOMEK, y_SMOTETOMEK, random_state=42, test_size= 0.20)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuraryXGBoostAndSMOTETomek = metrics.accuracy_score(y_test, y_pred)


    # 6. Save for label encoding view
    result2 = pd.concat([X, y], axis=1)

    # 7. SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    df_smote = pd.concat([pd.DataFrame(X_smote, columns=X.columns), pd.Series(y_smote, name="Outcome")], axis=1)
    result7 = df_smote.copy()

    # 8. SMOTETomek
    smote_tomek = SMOTETomek(random_state=42)
    X_smt, y_smt = smote_tomek.fit_resample(X, y)
    df_smote_tomek = pd.concat([pd.DataFrame(X_smt, columns=X.columns), pd.Series(y_smt, name="Outcome")], axis=1)
    result8 = df_smote_tomek.copy()

    # Train-Test split
    # def train_model(X, y):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     return accuracy_score(y_test, y_pred)

    # 9. Accuracy on each dataset
    # result3 = train_model(X_smote, y_smote)          # XGBoost + SMOTE
    # result4 = train_model(X, y)                      # XGBoost only
    # result5 = train_model(X_smt, y_smt)              # XGBoost + SMOTETomek

    # Dummy values for unused results (if needed later)
    result6 = None
    result9 = None
    result10 = None

    return (
        result1,     # Original dataset
        result2,     # Label encoded data (imputed & cleaned)
        accuraryXGBoostAndSMOTE,     # Accuracy: XGBoost + SMOTE
        accuraryXGBoost,     # Accuracy: XGBoost only
        accuraryXGBoostAndSMOTETomek,     # Accuracy: XGBoost + SMOTETomek
        result7,     # df_SMOTE
        result8,     # df_SMOTETomek
        result9,     # Placeholder
        result10     # Data after imputation and outlier removal
    )
