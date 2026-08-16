import json

import time

from pathlib import Path



import pandas as pd

from lightgbm import LGBMClassifier, early_stopping

from sklearn.metrics import (

    accuracy_score,

    f1_score,

    precision_score,

    recall_score,

    roc_auc_score,

)

from sklearn.model_selection import train_test_split





files = list((Path.home() / "ml-benchmark").glob("*.csv"))



if not files:

    raise FileNotFoundError(

        "Không tìm thấy file CSV trong ~/ml-benchmark"

    )



data_path = files[0]

print(f"Loading dataset: {data_path}")



start = time.perf_counter()

df = pd.read_csv(data_path)

load_time = time.perf_counter() - start



if "Class" not in df.columns:

    raise KeyError(

        f"Không tìm thấy cột 'Class'. Các cột hiện có: {list(df.columns)}"

    )



X = df.drop(columns=["Class"])

y = df["Class"]



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y,

)



model = LGBMClassifier(

    n_estimators=200,

    learning_rate=0.05,

    num_leaves=31,

    objective="binary",

    random_state=42,

    n_jobs=-1,

    verbosity=-1,

)



print("Training...")



start = time.perf_counter()



model.fit(

    X_train,

    y_train,

    eval_set=[(X_test, y_test)],

    eval_metric="auc",

    callbacks=[early_stopping(20, verbose=False)],

)



training_time = time.perf_counter() - start



probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= 0.5).astype(int)



one_row = X_test.iloc[[0]]



start = time.perf_counter()

model.predict_proba(one_row)

latency_ms = (time.perf_counter() - start) * 1000



batch = X_test.iloc[:1000]



start = time.perf_counter()

model.predict_proba(batch)

batch_time = time.perf_counter() - start



throughput = len(batch) / batch_time



result = {

    "data_file": str(data_path),

    "rows": int(len(df)),

    "load_data_seconds": round(load_time, 4),

    "training_seconds": round(training_time, 4),

    "best_iteration": int(model.best_iteration_),

    "auc_roc": round(

        roc_auc_score(y_test, probabilities),

        6,

    ),

    "accuracy": round(

        accuracy_score(y_test, predictions),

        6,

    ),

    "f1_score": round(

        f1_score(y_test, predictions, zero_division=0),

        6,

    ),

    "precision": round(

        precision_score(y_test, predictions, zero_division=0),

        6,

    ),

    "recall": round(

        recall_score(y_test, predictions, zero_division=0),

        6,

    ),

    "inference_latency_1_row_ms": round(latency_ms, 4),

    "inference_time_1000_rows_seconds": round(batch_time, 4),

    "inference_throughput_rows_per_second": round(

        throughput,

        2,

    ),

}



with open("benchmark_result.json", "w") as file:

    json.dump(result, file, indent=2)



print(json.dumps(result, indent=2))

print("Saved: benchmark_result.json")