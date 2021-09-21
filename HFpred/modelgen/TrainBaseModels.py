# Importing Libraries

# Import pandas library
import pandas as pd

# Import numpy library
import numpy as np

# Import Train-Test split library
from sklearn.linear_model import LogisticRegression

# Import RandomForestClassifier library
from sklearn.ensemble import RandomForestClassifier

# Import KNeighborsClassifier library
from sklearn.neighbors import KNeighborsClassifier

# Import Support Vector Machine (SVM) library
from sklearn import svm

# Import XGBClassifier library
import xgboost
import lightgbm
from xgboost import XGBClassifier

# Import GradientBoostingClassifier library
from sklearn.ensemble import GradientBoostingClassifier

#Import DecisionTreeClassifier library
from sklearn.tree import DecisionTreeClassifier

# Import Train-Test split library
from sklearn.model_selection import train_test_split

# Import KFold split library
from sklearn.model_selection import KFold

# Import accuracy score from computing library
from sklearn.metrics import accuracy_score

# Import confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Import ROC AUC score computing library
from sklearn.metrics import roc_auc_score

# Import TPR, FPR computing library
from sklearn.metrics import roc_curve

# Import matplotlib library
import matplotlib.pyplot as plt
#%matplotlib inline

# Import warnings
import warnings
warnings.filterwarnings('ignore')

# Import formating tools
from colorama import Fore, Back, Style
import seaborn as sns

# Import project specific libraries
from ProjectConstants import SPLIT_RATIO, RANDOM_STATE

"""
This class loads the final dataset, splits them into train-test, trains base models, and stores them.
"""

class TrainBaseModels():

    # init method or constructor
    def __init__(self):
        self.__load_source_data()

    def __load_source_data(self):
        self.df = pd.read_csv("data/output/HFdata_final.csv")

    def split_data(self):
        # Split the dataset into Train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df.drop('DEATH_EVENT', axis = 1),
                                self.df['DEATH_EVENT'], test_size=SPLIT_RATIO, random_state=RANDOM_STATE)

    # Logistic Regression model
    def train_logistic_regression(self):
        #Generate a Logistic Regression object
        lr_model = LogisticRegression()

        # Train a Logistic Regression model with Train dataset
        lr_model.fit(self.x_train, self.y_train)

        # Predict the outcome
        y_hat_lr = lr_model.predict(self.x_test)

        #compute the accuracy_score and print it
        accuracy_lr = accuracy_score(self.y_test, y_hat_lr)
        #print("Accuracy score of Logistic Regression model:", accuracy_lr)
        return accuracy_lr

        # Predict the AUC score and print it
        # y_hat_lr_proba = lr_model.predict_proba(self.x_test)
        # auc_lr = roc_auc_score(self.y_test, y_hat_lr_proba[:, 1])
        # print("AUC score of Logistic Regression Model:", auc_lr)
        # return auc_lr

    # Random Forest model
    def train_random_forest(self):

        # Generate a Random Forest Classifier object
        rf_model = RandomForestClassifier(min_samples_leaf=1, random_state = 2202)

        # Train a Random Forest model with Train dataset
        rf_model.fit(self.x_train, self.y_train)

        # Predict the outcome
        y_hat_rf = rf_model.predict(self.x_test)

        #compute the accuracy_score and print it
        accuracy_rf = accuracy_score(self.y_test, y_hat_rf)
        #print("Accuracy score of Random Forest model:", accuracy_rf)
        return accuracy_rf

        # Predict the AUC score and print it
        # y_hat_rf_proba = rf_model.predict_proba(self.x_test)
        # auc_rf = roc_auc_score(self.y_test, y_hat_rf_proba[:, 1])
        # print("AUC score of Random Forest Model:", auc_rf)
        # return auc_rf

    # K-Nearest Neighbor Model
    def train_knn_model(self):

        # Generate a k-Nearest Neighbor object
        knn_model = KNeighborsClassifier(n_neighbors=3)

        # Train a k-Nearest Neighbor model with Train dataset
        knn_model.fit(self.x_train, self.y_train)

        # Predict the outcome
        y_hat_knn = knn_model.predict(self.x_test)

        # Compute the accuracy score and print it
        accuracy_knn = accuracy_score(self.y_test, y_hat_knn)
        #print("Accuracy score of K-Nearest Neighbor model:", accuracy_knn)
        return accuracy_knn

        # Predict the AUC score and print it
        # y_hat_knn_proba = knn_model.predict_proba(self.x_test)
        # auc_knn = roc_auc_score(self.y_test, y_hat_knn_proba[:, 1])
        # print("AUC score of K-Nearest Neighbor Model:", auc_knn)
        # return auc_knn

    # Generate a Support Vector Machine (SVM) Model
    def train_svm_model(self):

        # Generate a Support Vector Machine (SVM) object
        svm_model = svm.SVC(kernel='linear')

        # Train a Support Vector Machine model with Train dataset
        svm_model.fit(self.x_train, self.y_train)

        # Predict the outcome
        y_hat_svm = svm_model.predict(self.x_test)

        # Compute the accuracy score and print it
        accuracy_svm = accuracy_score(self.y_test, y_hat_svm)
        #print("Accuracy score of Support Vector Machine model:", accuracy_svm)
        return accuracy_svm

        # Predict the AUC score and print it
        # Generate a Support Vector Machine (SVM) object to determine probability of binary outcome
        # svm_model_proba = svm.SVC(kernel='linear', probability=True)
        # svm_model_proba.fit(self.x_train, self.y_train)
        # y_hat_svm_proba = svm_model_proba.predict_proba(self.x_test)
        # auc_svm = roc_auc_score(self.y_test, y_hat_svm_proba[:, 1])
        #print("AUC score of Support Vector Machine Model:", auc_svm)
        #return auc_svm

    # Generate a XGBoost Classifier Model
    def train_xgb_model(self):

        # Generate a XGBoost object
        xgb_model = XGBClassifier(learning_rate =0.01,
                      subsample=0.75,
                      colsample_bytree=0.72,
                      min_child_weight=8,
                      max_depth=5)

        # Train a XGBoost model with Train dataset
        xgb_model.fit(self.x_train, self.y_train)

        # Predict the outcome
        y_hat_xgb = xgb_model.predict(self.x_test)

        # Compute the accuracy score and print it
        accuracy_xgb = accuracy_score(self.y_test, y_hat_xgb)
        #print("Accuracy score of XGBoost model:", accuracy_xgb)
        return accuracy_xgb

        # Predict the AUC score and print it
        # y_hat_xgb_proba = xgb_model.predict_proba(self.x_test)
        # auc_xgb = roc_auc_score(self.y_test, y_hat_xgb_proba[:, 1])
        # print("AUC score of XGBoost Model:", auc_xgb)
        # return auc_xgb

    # Generate a Decision Tree Classifier Model
    def train_dtc_model(self):

        # Generate a Decision Tree Classifier object
        dtc_model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')

        # Train a XGBoost model with Train dataset
        dtc_model.fit(self.x_train, self.y_train)

        # Predict the outcome
        y_hat_dtc = dtc_model.predict(self.x_test)

        # Compute the accuracy score and print it
        accuracy_dtc = accuracy_score(self.y_test, y_hat_dtc)
        #print("Accuracy score of Decision Tree Classifier model:", accuracy_dtc)
        return accuracy_dtc

        # Predict the AUC score and print it
        # y_hat_dtc_proba = dtc_model.predict_proba(self.x_test)
        # auc_dtc = roc_auc_score(self.y_test, y_hat_dtc_proba[:, 1])
        #print("AUC score of Decision Tree Classifier Model:", auc_xgb)
        #return auc_dtc

    # Generate GradientBoostingClassifier model
    def train_gbc_model(self):

        # Generate a Gradient Boosting Classifier object
        gbc_model = GradientBoostingClassifier(max_depth=2, random_state=1)

        # Train GBC model with Train dataset
        gbc_model.fit(self.x_train, self.y_train)

        # Predict the outcome
        y_hat_gbc = gbc_model.predict(self.x_test)

        # Compute the accuracy score and print it
        accuracy_gbc = accuracy_score(self.y_test, y_hat_gbc)
        #print("Accuracy score of Gradient Boosting Classifier model:", accuracy_gbc)
        return accuracy_gbc

        # Predict the AUC score and print it
        # y_hat_gbc_proba = gbc_model.predict_proba(self.x_test)
        # auc_gbc = roc_auc_score(self.y_test, y_hat_gbc_proba[:, 1])
        # print("AUC score of Gradient Boosting Classifier Model:", auc_gbc)
        # return auc_gbc

# Class ends here

def export_performance_metrics(accuracy_lr,accuracy_rf, accuracy_knn, accuracy_svm, accuracy_xgb,accuracy_dtc, accuracy_gbc):
    perf_metrics_df = pd.DataFrame({'Metrics:': ['Accuracy:'],
                                    'Logistic Regression': [accuracy_lr],
                                    'Random Forest': [accuracy_rf],
                                    'K-Nearest Neighbor': [accuracy_knn],
                                    'Support Vector Machine': [accuracy_svm],
                                    'XGBoost': [accuracy_xgb],
                                    'Decision Tree Classifier': [accuracy_dtc],
                                    'Gradient Boosting Classifier': [accuracy_gbc]})

    perf_metrics_df.to_csv("data/output/baseline_metrics.csv", index=False)

def main():
    base_models = TrainBaseModels()
    base_models.split_data()
    accuracy_lr = base_models.train_logistic_regression()
    #auc_lr = base_models.train_logistic_regression()
    accuracy_rf = base_models.train_random_forest()
    #auc_rf = base_models.train_random_forest()
    accuracy_knn = base_models.train_knn_model()
    #auc_knn = base_models.train_knn_model()
    accuracy_svm = base_models.train_svm_model()
    #auc_svm = base_models.train_svm_model()
    accuracy_xgb = base_models.train_xgb_model()
    #auc_xgb = base_models.train_xgb_model()
    accuracy_dtc = base_models.train_dtc_model()
    #auc_dtc = base_models.train_dtc_model()
    accuracy_gbc = base_models.train_gbc_model()
    #auc_gbc = base_models.train_gbc_model()

    export_performance_metrics(accuracy_lr,accuracy_rf, accuracy_knn, accuracy_svm, accuracy_xgb,accuracy_dtc, accuracy_gbc)

if __name__ == "__main__":
    main()
