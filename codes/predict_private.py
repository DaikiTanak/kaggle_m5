from datetime import datetime, timedelta
import time
import gc
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

import create_feature
import load_data

def predict_private(lag_win_pairs, model_path):

    start_time = time.time()

    path_data = "../input/m5-forecasting-accuracy/test_basic.csv"
    te = pd.read_csv(path_data, index_col=0)
    te = load_data.reduce_mem_usage(te)

    cols = [f"F{i}" for i in range(1,29)]
    first_day = datetime(2016, 5, 23)

    m_lgb = lgb.Booster(model_file=model_path)
    print(f"LGBM model loaded: {model_path}")

    useless_cols = ["id", "date", "sales","d",
                    "wm_yr_wk", "weekday",
                   "state_name", "snap_CA", "snap_TX", "snap_WI"]



    max_lags = 100
    alpha = 1.0

    for tdelta in range(0, 28):
        start = time.time()
        # target date to predict sales
        day = first_day + timedelta(days=tdelta)
        print("Predicting : ", tdelta, day.date())

        # target period
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()

        create_feature.create_fea(tst, lag_win_pairs=lag_win_pairs)

        train_cols = tst.columns[~tst.columns.isin(useless_cols)]

        tst = tst.loc[tst.date == day, train_cols]
        # fill "sales" with predicted value by model
        predictions = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev
        te.loc[te.date == day, "sales"] = predictions
        print(f"Elapsed time: {(time.time() - start) // 60} min.")

    te_sub = te.loc[te.date >= first_day, ["id", "sales"]].copy()
    # del te; gc.collect()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0., inplace=True)
    te_sub.sort_values("id", inplace=True)
    te_sub.reset_index(drop=True, inplace=True)

    sub2 = te_sub.copy()
    sub2["id"] = sub2["id"].str.replace("evaluation$", "validation")
    sub = pd.concat([sub2, te_sub], axis=0, sort=False)
    sub.to_csv(f"../result/private_predictions.csv",index=False)

    print(f"Inference time: {(time.time()-start_time) // 60} min.")

def test_private_prediction():
    model_path = "../result/lgbm-train_56bd169ef07c4811acee5bcf2417c35b.model"
    lag_win_pairs = None
    predict_private(lag_win_pairs, model_path)


if __name__ == "__main__":
    # test_private_prediction()
    predict_private(lag_win_pairs=[[28,28],[7,7],[3,3],[1,4],[1,7],[1,14],[1,28]],
                    model_path="../result/retrain.model")
