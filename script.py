
import argparse
import joblib
import os

import numpy as np
import pandas as pd
import sklearn
import boto3
import pathlib
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")
    # parser.add_argument(
    #     "--features", type=str
    # )  # in this script we ask user to explicitly name features
    # parser.add_argument(
    #     "--target", type=str
    # )  # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    # print("====================== args.model_di =================")
    # print(args.model_dir)
    # print("====================== args.model_di =================")

    print("reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    features = list(train_df.columns)
    label = features.pop(-1)

    print("building training and testing datasets")
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print(f"Features: {features}")
    print(f"Label: {label}")

    print("After  train test split")


    try:
        os.makedirs(args.model_dir, exist_ok=True)
        print(f"Directory created: {args.model_dir}")
    except Exception as e:
        print(f"Error creating model directory: {str(e)}")
        raise
    
    print("====================== args.model_di =================")
    print(args.model_dir)
    print("====================== args.model_di =================")

    # train
    print("training model")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.random_state
    )

    model.fit(X_train, y_train)

    # # print abs error
    # # print("validating model")
    # # abs_err = np.abs(model.predict(X_test) - y_test)

    # # print couple perf metrics
    # for q in [10, 50, 90]:
    #     print("AE-at-" + str(q) + "th-percentile: " + str(np.percentile(a=abs_err, q=q)))

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)
    # print(args.min_samples_leaf)

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print("Testing Accuracy: ", test_acc)
    print("Testing Report: ", test_rep)

