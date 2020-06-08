import pandas as pd
import numpy as np


def create_fea(dt, cash_df=None, start_date=None) -> None:
    # start_date: start date to create features from.

    useless_cols = []

#     lags = [7, 28]
    lags = [2, 5, 7, 28]

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

#     wins = [7, 28]
        wins = [3, 7, 14, 28]
    for win in wins :
        for lag, lag_col in zip(lag_saver, lag_cols):
#             print(f"Making rolling features. lag_col:{lag_col} lag:{lag} window:{win}")
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

#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")

    for col in useless_cols:
        del dt[col]
    gc.collect()
