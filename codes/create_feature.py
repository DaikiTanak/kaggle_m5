import pandas as pd
import numpy as np
import gc

def roll_target(lag_size: int, window_size: int, categorical_feature: str):
    """
        Roll target value w.r.t. category
        Params:
            lag_size: lag size for target
            window_size: window size for rolling
            categorical_feature: feature name to roll about
    """
    return

def create_fea(dt, cash_df=None, start_date=None) -> None:
    # start_date: start date to create features from.

    print("Creating features...")

    useless_cols = []


    lag_win_pairs = [
        (28, 28),
        (28, 7),
        # (7,7)
    ]
    lags = list(set(map(lambda x: x[0], lag_win_pairs)))
    wins = list(set(map(lambda x: x[1], lag_win_pairs)))

    lag_about = ["id","item_id", "dept_id", "cat_id", "store_id", "state_id"]
    today_sale_cols = []

    for col in lag_about:
        if col == "id":
            continue
        dt[f"d_{col}"] = dt["d"].astype(str) + "_" + dt[col].astype(str)
        dt[f"today_{col}_mean"] = dt[["sales", f"d_{col}"]].groupby([f"d_{col}"])["sales"].transform("mean")
        today_sale_cols.append(f"today_{col}_mean")

        useless_cols.append(f"today_{col}_mean"); useless_cols.append(f"d_{col}");

    lag_cols = []
    lag_saver = []
    lag_features = []

    for lag in lags:
        for feature in lag_about:
            lag_cols.append(f"salelag_{lag}_{feature}")
            lag_saver.append(lag)
            lag_features.append(feature)

    for lag, lag_col, f in zip(lag_saver, lag_cols, lag_features):
        if f == "id":
            dt[lag_col] = dt[["id", "sales"]].groupby("id")["sales"].shift(lag)
        else:
            dt[lag_col] = dt[["id", f"today_{f}_mean"]].groupby("id")[f"today_{f}_mean"].shift(lag)



    for (lag, win) in lag_win_pairs:
        for lag_col in lag_cols:

            dt[f"rmean_{lag_col}_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
#             dt[f"rmedian_{lag_col}_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).median())

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }


    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")

    for col in useless_cols:
        del dt[col]
    gc.collect()

    return
