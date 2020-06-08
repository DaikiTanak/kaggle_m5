from datetime import datetime, timedelta
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import swifter # quick pandas.apply
pd.options.display.max_columns = 50


CAL_DTYPES={
    "event_name_1": "category",
    "event_name_2": "category",
    "event_type_1": "category",
    "event_type_2": "category",
    "weekday": "category",
    'wm_yr_wk': 'int16',
    "wday": "int8",
    "month": "int8",
    "year": "int16",
    "snap_CA": "int8",
    'snap_TX': 'int8',
    'snap_WI': 'int8' }

PRICE_DTYPES = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price":"float32" }

FIRST_DAY = 1 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
h = 28
max_lags = 60
tr_last = 1941

# start date where public lb and private lb are calculated
fday = datetime(2016,4, 25)
seed = 46

dev_firstdate = fday - timedelta(tr_last - FIRST_DAY)

def create_dt(PATH_PRICE_CSV, PATH_CALENDER_CSV, PATH_SALES_CSV, is_train=True, nrows=None, first_day=1200, ):

    # load item-price csv.
    prices = pd.read_csv(PATH_PRICE_CSV, dtype=PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    # load calender csv.
    cal = pd.read_csv(PATH_CALENDER_CSV, dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()

    # load sales csv.
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols}
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(PATH_SALES_CSV,
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)

    dt["state_name"] = dt["state_id"].copy()

    # cat2id
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()

    if not is_train:
        # test dataframe
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan

    dt = pd.melt(dt,
                  id_vars = catcols+["state_name"],
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")

    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

    dt["date"] = pd.to_datetime(dt["date"], format="%Y-%m-%d")

    # create snap_flag feature
    def make_snap_flag(row):
        state_name = row["state_name"]
        return row[f"snap_{state_name}"]
    dt["snap_flag"] = dt.swifter.apply(make_snap_flag, axis=1)
    del dt["state_name"]; gc.collect();

    return dt


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type)[:8] != "category" and str(col_type)[:8] != "datetime":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
#             df[col] = df[col].astype('category')
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    return df



def save_basic_df():
    cwd = ""
    PATH_PRICE_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/sell_prices.csv")
    PATH_CALENDER_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/calendar.csv")
    # PATH_SALES_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/sales_train_validation.csv")
    PATH_SALES_CSV = os.path.join(cwd, "../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
    df = load_data.create_dt(PATH_PRICE_CSV, PATH_CALENDER_CSV, PATH_SALES_CSV, first_day=1,)
    df.to_csv("../input/m5-forecasting-accuracy/sales_train_evaluation_basic.csv")
    return


if __name__ == "__main__":
    save_basic_df()
