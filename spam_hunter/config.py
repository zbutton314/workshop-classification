# File directories
data_path_train = "data/spambase.csv"
data_path_val = "data/spambase_val.csv"
model_results_path = "model_results.json"
model_path = "best_model.pkl"

# Hyperparameter Specification
params = {
    "nb": {
        "spec": {},
        "tune": {}
    },
    "knn": {
        "spec": {},
        "tune": {
            "n_neighbors": [1, 2, 3, 4, 5]
        }
    },
    "lr": {
        "spec": {
            "solver": "liblinear"
        },
        "tune": {
            "penalty": ["l1", "l2"]
        }
    },
    "dt": {
        "spec": {},
        "tune": {
            "max_depth": [1, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [2, 5, 10]
        }
    },
    "rf": {
        "spec": {},
        "tune": {
            "max_depth": [1, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [2, 5, 10]
        }
    },
    "xgb": {
        "spec": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        },
        "tune": {
            "max_depth": [1, 5, 10],
            "colsample_bytree": [0.5, 1.0],
            "reg_alpha": [0.1, 1.0]
        }
    },
    "svc": {
        "spec": {
            "probability": True,
            "random_state": 314
        },
        "tune": {
            "kernel": ["linear", "poly", "rbf", "sigmoid"]
        }
    }
}


