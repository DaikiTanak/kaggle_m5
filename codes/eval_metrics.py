## importing packages
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


from typing import Union
from tqdm.notebook import tqdm

## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get scores at all levels
class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'cat_id',
            'state_id',
            'dept_id',
            'store_id',
            'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        group_ids = []
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores


def get_public_score(prediction_df):

    df_train_full = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
    ## reading data
    df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
    df_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
    df_sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
    df_sample_submission["order"] = range(df_sample_submission.shape[0])

    df_train = df_train_full.iloc[:, :-28]
    df_valid = df_train_full.iloc[:, -28:]

    evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)

    ## evaluating random submission
    preds_valid = pd.read_csv("../result/sub.csv")
    preds_valid = preds_valid[preds_valid.id.str.contains("validation")]
    preds_valid = preds_valid.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1).reset_index(drop = True)
    preds_valid.rename(columns = {
        "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",
        "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",
        "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",
        "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"
    }, inplace = True)
    groups, scores = evaluator.score(preds_valid)

    score_public_lb = np.mean(scores)

    for i in range(len(groups)):
        print(f"Score for group {groups[i]}: {round(scores[i], 5)}")

    print(f"\nPublic LB Score: {round(score_public_lb, 5)}")


#
# weight_mat = np.c_[np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
#                    pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
#                    pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values,
#                    np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
#                    ].T

# weight_mat_csr = csr_matrix(weight_mat)
# del weight_mat; gc.collect()
#
# def weight_calc(data, product):
#
#     # calculate the denominator of RMSSE, and calculate the weight base on sales amount
#
#     sales_train_val = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
#
#     d_name = ['d_' + str(i+1) for i in range(1913)]
#
#     sales_train_val = weight_mat_csr * sales_train_val[d_name].values
#
#     # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
#     # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
#     df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))
#
#     start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
#
#     flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1
#
#     sales_train_val = np.where(flag,np.nan,sales_train_val)
#
#     # denominator of RMSSE / RMSSEの分母
#     weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1913-start_no)
#
#     # calculate the sales amount for each item/level
#     df_tmp = data[(data['date'] >= '2016-03-25') & (data['date'] <= '2016-04-24')]
#     df_tmp['amount'] = df_tmp['sales'] * df_tmp['sell_price']
#     df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum)
#     df_tmp = df_tmp[product.id].values
#
#     weight2 = weight_mat_csr * df_tmp
#
#     weight2 = weight2/np.sum(weight2)
#
#     del sales_train_val
#     gc.collect()
#
#     return weight1, weight2
#
# weight1, weight2 = weight_calc(df, product)
#
# DAYS_PRED = 28
# def wrmsse(preds, data):
#
#     # this function is calculate for last 28 days to consider the non-zero demand period
#     # actual obserbed values / 正解ラベル
#     y_true = data.get_label()
#
#     y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
#     preds = preds[-(NUM_ITEMS * DAYS_PRED):]
#     # number of columns
#     num_col = DAYS_PRED
#
#     # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
#     reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
#     reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
#
#
#     train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
#
#     score = np.sum(
#                 np.sqrt(
#                     np.mean(
#                         np.square(
#                             train[:,:num_col] - train[:,num_col:])
#                         ,axis=1) / weight1) * weight2)
#
#     return 'wrmsse', score, False
