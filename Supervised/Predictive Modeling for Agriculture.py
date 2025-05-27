# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Write your code here
crops.head()

X = crops.drop(["crop"], axis=1)
y = crops["crop"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

features_dict = {}
feature_performance = {}

for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression( multi_class = "multinomial" )
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    feature_performance[feature] = metrics.f1_score(y_test, y_pred, average="weighted")
    print(f"F1-score for {feature}: {feature_performance}")


key, val = list(feature_performance.items())[2]

best_predictive_feature = {key: val}
print(best_predictive_feature)