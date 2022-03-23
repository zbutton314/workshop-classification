import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
import json
import logging
import sys
import warnings

from spam_hunter.config import *
from spam_hunter.utils import *


logger = logging.getLogger(__name__)
logging.basicConfig(stream = sys.stdout, 
                    level = logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess_data(data_path):
  """Import data and normalize all independent variables
    Args:
      data_path: Location of data file
    Returns:
      X_ss: Pandas data frame with preprocessed independent variables
      y: Pandas series with preprocessed dependent variable
  """
  logger.info("Importing and Preprocessing Data")
  c.start("pd")

  df = pd.read_csv(data_path, header=0)
  df = df.rename(columns={"char_freq_[": "char_freq_sqbracket"})
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]
  ss = StandardScaler()
  ss.fit(X)
  X_ss = pd.DataFrame(ss.transform(X))
  X_ss.columns = X.columns

  logger.info(f"-- Preprocessing Complete: {c.stop('pd')} seconds")

  return X_ss, y


def initialize_models():
  """Instantiate model objects with fixed hyperparameters
    Returns:
      List of tuples - ("model_key", model_object)
  """
  logger.info("-- Creating Model Objects")
  c.start("cm")

  nb_model = GaussianNB()
  knn_model = KNeighborsClassifier()
  lr_model = LogisticRegression(**params["lr"]["spec"])
  dt_model = DecisionTreeClassifier()
  rf_model = RandomForestClassifier()
  xgb_model = XGBClassifier(**params["xgb"]["spec"])
  svc_model = SVC(**params["svc"]["spec"])

  nb = ("nb", nb_model)
  knn = ("knn", knn_model)
  lr = ("lr", lr_model)
  dt = ("dt", dt_model)
  rf = ("rf", rf_model)
  xgb = ("xgb", xgb_model)
  svc = ("svc", svc_model)

  logger.info(f"---- Model Initialization Complete: {c.stop('cm')} seconds")

  return [nb, knn, lr, dt, rf, xgb, svc]


def run_grid_search(X, y, model, params, scoring="f1", n_folds=10):
  """Perform grid search with cross validation using specified data, model, and parameters
    Args:
      X: Pandas data frame with independent variables
      y: Pandas series with dependent variable
      model: Initialized model object
      params: Dict with hyperparameters to search
      scoring: Metric to optimize
      n_folds: Number of folds for K-Fold CV
    Returns:
      best_model: Fitted model object with highest performance
      best_params: Params for model with highest performance
      best_score: Highest score for this model type
  """
  cv = GridSearchCV(model, params, scoring=scoring, cv=n_folds)
  cv.fit(X, y)
  best_model = cv.best_estimator_
  best_params = cv.best_params_
  best_score = cv.best_score_

  return best_model, best_params, best_score


def train_models(X, y):
  """Perform hyperparameter tuning for each model
    Args:
      X: Pandas data frame with independent variables
      y: Pandas series with dependent variables
    Returns:
      model_results: Dict with params and score for best model of each type
      best_model: Fitted model object with highest overall performance
  """
  logger.info("Training Models")
  c.start("tm")

  model_list = initialize_models()
  model_results = {}
  best_score = 0

  for model_tup in model_list:
    model_key = model_tup[0]
    logger.info(f"---- Running Grid Search ({model_key})")
    c.start("gs")

    params_tune = params[model_key]["tune"]
    model, model_params, model_score = run_grid_search(X=X, y=y, model=model_tup[1], 
                                                                  params=params_tune, scoring="f1", n_folds=10)
    model_results[model_key] = {}
    model_results[model_key]["params"] = model_params
    model_results[model_key]["score"] = model_score

    if model_score > best_score:
      best_score = model_score
      best_model = model

    logger.info(f"------ Grid Search Complete ({model_key}): {c.stop('gs')} seconds")

  logger.info(f"-- Model Training Complete: {c.stop('tm')} seconds")

  return model_results, best_model


def export_results(results, model):
  """Export results dictionary and model object to local directory
    Args:
      results: Dict with modeling results
      model: Fitted model object with highest overall performance
  """
  logger.info("Exporting Model and Results")
  c.start("er")

  with open(model_results_path, "w") as path_out:
    json.dump(results, path_out)

  joblib.dump(model, model_path)

  logger.info(f"-- Export Complete: {c.stop('er')} seconds")


def validate_model(model_path, data_path):
  """Use model to predict for validation set and print results
    Args:
      model_path: Location of model object
      data_path: Location of data file
  """
  logger.info("Running Model Validation")
  c.start("mv")

  model = joblib.load(model_path)

  X_val, y_val = preprocess_data(data_path=data_path)
  y_pred = model.predict(X_val)
  print(metrics.classification_report(y_val, y_pred))
  print(f"Accuracy: {round(100*metrics.accuracy_score(y_val, y_pred), 1)}%")
  print(f"F1: {round(100*metrics.f1_score(y_val, y_pred), 1)}%")

  logger.info(f"-- Validation Complete: {c.stop('mv')} seconds")
  

if __name__ == "__main__":
  X, y = preprocess_data(data_path=data_path_train)
  model_results, best_model = train_models(X=X, y=y)
  export_results(results=model_results, model=best_model)
  validate_model(model_path=model_path, data_path=data_path_val)
