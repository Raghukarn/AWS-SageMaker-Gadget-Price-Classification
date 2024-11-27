# %%
# pip install sagemaker
# !pip install scikit-learn

# %%
import sagemaker
from sklearn.model_selection import train_test_split
import boto3
import pandas as pd

# %%
sm_boto3 = boto3.client("sagemaker")
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = "mobpricebucketsagemaker2024"
print("Using bucket: " + bucket)

# %%
df = pd.read_csv("train.csv")

# %%
df.head()

# %%
df.shape

# %%
df['price_range'].value_counts(normalize=True)

# %%
df.columns

# %%
# % of values are misssing
df.isnull().mean() * 100

# %%
features = list(df.columns)
features

# %%
label = features.pop(-1)
label

# %%
x = df[features]
y = df[label]

# %%
x.head()

# %%
y.head()

# %%
x.shape, y.shape

# %%
y.value_counts()

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
#  convert to dataframe

trainX = pd.DataFrame(X_train)
trainX[label] = y_train

testX = pd.DataFrame(X_test)
testX[label] =  y_test

# %%
trainX

# %%
testX

# %%
#  let's create a local version of these data frames so that we can push them to S3 bucket later

trainX.to_csv("train-V-1.csv", index = False)
testX.to_csv("test-V-1.csv", index  = False)

# %%
# push data to S3 and SageMaker will train data from S3

sk_prefix = "sagemaker/mobile_price_classification/sklearncontainer"

trainpath = sess.upload_data(
    path = "train-V-1.csv", bucket=bucket, key_prefix=sk_prefix
)

testpath = sess.upload_data(
    path = "test-V-1.csv", bucket=bucket, key_prefix=sk_prefix
)

print(trainpath)
print(testpath)

# %% [markdown]
# so far we done the data ingestion steps. </br>
# Let's follow sagemaker

# %%
%%writefile script.py

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



# %%
from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VESION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role="arn:aws:iam::211125616305:role/sagemakerrole",
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version=FRAMEWORK_VESION,
    base_job_name="RF-custom-sklearn",
    hyperparameters={
        "n_estimators":100,
        "random_state":0
    },
    use_spot_instances = True,
    max_wait = 7200,
    max_run = 3600

)

# %%


# %%
# launch training job, with asynchronous call
sklearn_estimator.fit({"train":trainpath, "test":testpath}, wait=True)

# %%
sklearn_estimator.latest_training_job.wait(logs="None")
artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model Artifacts persisted at " + artifact)

# %%
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime

model_name = "Custom-sklearn-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model = SKLearnModel(
    name = model_name,
    model_data=artifact,
    role="arn:aws:iam::211125616305:role/sagemakerrole",
    entry_point="script.py",
    framework_version=FRAMEWORK_VESION
)

# %%
model

# %% [markdown]
# we created an additional copy of our model in S3 and use that particular copy in a new folder (custom-sklearn-model) to deploy the model to be used for end point. Just to be more clear on folders to avoid any ambiguity bqz training can be dne many times and each training will create a folder and respective model.

# %%
#  depoy this specific model as an end point

endpoint_name = "Custom-sklearn-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndPointName={}".format(endpoint_name))

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name
)

# %%
# we can use this predictor to use for any classification and even its return type itself is SKLearnPredictor
predictor

# %%
endpoint_name

# %% [markdown]
# Let's try to sample data to classify

# %%
# take sample data
testX[features][0:2].values.tolist()

# %%
print(predictor.predict(testX[features][0:2].values.tolist()))

# %%
sm_boto3.delete_endpoint(EndpointName=endpoint_name)

# %%



