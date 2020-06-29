from datetime import datetime, timedelta
import time
import gc
import os
import numpy as np
import pandas as pd


# User defined func
import eval_metrics
import load_data
import create_feature
from mlflow_writer import MlflowWriter

from omegaconf import DictConfig, ListConfig
import hydra
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
import mlflow

def date2d(date):
    dev_lastdate = datetime(2016,4, 24)

    tr_last = 1941
    diff = (dev_lastdate -date).days
    return tr_last - diff

# TODO: implemente retrain option.
def train_lgbm(df, cfg, retrain=False):
    # train_validation split

    cwd = hydra.utils.get_original_cwd()

    df_calendar = pd.read_csv(os.path.join(cwd,"../input/m5-forecasting-accuracy/calendar.csv"))
    df_prices = pd.read_csv(os.path.join(cwd,"../input/m5-forecasting-accuracy/sell_prices.csv"))

    df_sales = pd.read_csv(os.path.join(cwd,"../input/m5-forecasting-accuracy/sales_train_evaluation.csv"))


    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    cat_feats.extend(["event_name_1", "event_name_2",
                      "event_type_1", "event_type_2",
                      "wday", "month", "year",
                      "snap_flag"])

    useless_cols = ["id", "date", "sales","d",
                    "wm_yr_wk", "weekday",
                   "state_name", "snap_CA", "snap_TX", "snap_WI"]


    if cfg.lgbm.optuna_tuning:
        import optuna.integration.lightgbm as lgb
    else:
        import lightgbm as lgb

    fold_val_scores = dict()


    """
    2016/3/24 ~ 2016/4/24 : validation set
    ~ 2016/3/24 : training set
    """

    for fold_idx in range(1, 1+n_folds, 1):
        print("*"*20)
        print(f"fold {fold_idx}...")
        print("*"*20)

        val_firstdate = dev_lastdate - timedelta(days=val_days*fold_idx)
        val_lastdate = dev_lastdate - timedelta(days=val_days*(fold_idx-1))
        train_lastdate = val_firstdate-timedelta(1)

        print("train period:", dev_firstdate.date(), "~", train_lastdate.date())
        print("validation period:", val_firstdate.date(), "~", val_lastdate.date())

        train_df = df.query("date < @val_firstdate")
        val_df = df.query("@val_lastdate >= date > @train_lastdate")

        val_df_wrmsse = df_sales.iloc[:, -28:]

        # wrmsse_evaluator = eval_metrics.WRMSSEEvaluator(df_sales.iloc[:, :-28],
        #                                                 val_df_wrmsse,
        #                                                 calendar=df_calendar,
        #                                                 prices=df_prices,
        #                                                 val_firstdate=date2d(val_firstdate),
        #                                                 val_lastdate=date2d(val_lastdate),
        #                                                 converted_val_df=val_df,
        #                                                 )

        del df; gc.collect();

        print(min(train_df["date"]), max(train_df["date"]))
        print(min(val_df["date"]), max(val_df["date"]))

        train_df[:500].dropna(inplace=True)
        train_cols = train_df.columns[~train_df.columns.isin(useless_cols)]

        train_data = lgb.Dataset(train_df[train_cols],
                                 label=train_df["sales"],
                                 free_raw_data=False)
        val_data = lgb.Dataset(val_df[train_cols],
                                 label=val_df["sales"],
                                 free_raw_data=False)

        del train_df; gc.collect()

        lgbm_params = {}
        for k, v in cfg.lgbm.model_params.items():
            if isinstance(v, ListConfig):
                lgbm_params[k] = list(v)
            else:
                lgbm_params[k] = v
        print(lgbm_params)


        m_lgb = lgb.train(lgbm_params,
                          train_data,
                          # valid_sets=[train_data, val_data],
                          valid_sets=[val_data],
                          num_boost_round=cfg.lgbm.train_params.num_boost_round,
                          early_stopping_rounds=cfg.lgbm.train_params.early_stopping_rounds,
                          categorical_feature=cat_feats,
                          verbose_eval=10,)
                          # feval=wrmsse_evaluator.wrmsse_metric_lgbm)

        m_lgb.save_model(os.path.join(cwd, f"../result/fold{fold_idx}.lgb"))

        val_pred = m_lgb.predict(val_df[train_cols].values, num_iteration=m_lgb.best_iteration)


        # _, val_score, _ = wrmsse_evaluator.wrmsse_metric_lgbm(val_pred, val_df[train_cols])
        # print(f"VAL WRMSSE:{val_score}")

        # fold_val_scores[fold_idx] = val_score

        del val_df; gc.collect()

    #     m_lgb.save_model(f"../result/targetencoding_fullmodel_fold{fold_idx}.lgb")
        # model_savepath = os.path.join(hydra.utils.get_original_cwd(), "../result/no_fe_fold{fold_idx}.lgb")
        # m_lgb.save_model(model_savepath)

    importance = pd.DataFrame(m_lgb.feature_importance(), index=train_cols, columns=['importance']).sort_values("importance",inplace=False,ascending=False)

    # importance.to_csv("")

    return m_lgb, fold_val_scores, train_cols



@hydra.main(config_path='../config/config.yaml')
def run(cfg: DictConfig,):
    cwd = hydra.utils.get_original_cwd()


    # lag_win_pairs = [
    #         (28, 28),
    #         (28, 7),
    #         (7,7),
    #         (7, 3),
    #         (1, 3),
    #     ]
    lag_win_pairs = cfg.feat.lag_win_pairs

    writer.log_params_from_omegaconf_dict(cfg)
    print(os.getcwd())
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))


    # ----------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------
    PATH_PRICE_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/sell_prices.csv")
    PATH_CALENDER_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/calendar.csv")
    # PATH_SALES_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/sales_train_validation.csv")
    PATH_SALES_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
    PATH_SAMPLE_SUB_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/sample_submission.csv")

    load_start_time = time.time()
    if cfg.data.path_basic_df == "":
        print("making basic df from scratch...")


        df = load_data.create_dt(PATH_PRICE_CSV, PATH_CALENDER_CSV, PATH_SALES_CSV, first_day=1500,)
        df = load_data.reduce_mem_usage(df)
    else:
        print("loading saved basic csv...")
        path_df = cfg.data.path_basic_df
        df = pd.read_csv(os.path.join(cwd, path_df), index_col=0)
        df = load_data.reduce_mem_usage(df)
        print(df.shape)

    print(f"data loading time:{(time.time() - load_start_time)//60} min.")

    df = df.query("date > @dev_firstdate")
    gc.collect()

    # ----------------------------------------------------------------------
    # feature engineering
    # ----------------------------------------------------------------------

    # max_lags = max(list(map(lambda x:x[0], lag_win_pairs))) * 2
    max_lags = 56+29
    create_feature.create_fea(df, lag_win_pairs=lag_win_pairs)

    # ----------------------------------------------------------------------
    # Train model
    # ----------------------------------------------------------------------
    if cfg.lgbm.pretrained:
        m_lgb = lgb.Booster(model_file=os.path.join(cwd, cfg.lgbm.pretrained_model_path))

    else:
        model, _, train_cols = train_lgbm(df, cfg)
        model_savepath = os.path.join(cwd, f"../result/retrain.model")
        model.save_model(model_savepath)
        writer.log_param("model_path", model_savepath)


if __name__ == "__main__":

    def d2date(d:int) -> datetime:
        # convert d_** into datetime object.
        # datetime(2016, 6, 19) is d_1969.
        return datetime(2016,6,19) - timedelta(1969 - d)


    val_days = 27
    n_folds = 1

    dev_last = 1913
    dev_first = 1
    dev_firstdate = d2date(dev_first)
    # used for lgbm-training and validation.
    dev_lastdate = d2date(1941) # d_1913
    # public_firstdate = d2date(1914) # d_1914
    # public_lastdate = datetime(2016, 5, 22) # d_1941

    # private is for d_1941 ~ d_1969
    # private_lastdate = datetime(2016, 6, 19) # d_1969
    # print(f"dev_first:{dev_firstdate} dev_last:{dev_lastdate} public_last:{public_lastdate}")


    experiment_name = "lgbm-retrain"
    task_name = "test"
    writer = MlflowWriter(experiment_name, task_name)

    start_time = time.time()
    run()
    elapsed_time = time.time() - start_time
    writer.log_metric("elapsed_time_min", elapsed_time//60)
