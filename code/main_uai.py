# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table like and matrices
import pandas as pd
import numpy as np

# system
import os
import time

# self-defined model
from feature_uai import *

# Modeling
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import xgboost as xgb

cache_path = '../../UAI_data/output/cache'

poi_path = '../../UAI_data/input/poi_re.csv'
train_Aug_path = '../../UAI_data/input/train_Aug_re.csv'
train_Jul_path = '../../UAI_data/input/train_July_re.csv'
train_jul_demand_path = '../../UAI_data/input/tain_jul_demand.csv'
train_aug_demand_path = '../../UAI_data/input/tain_aug_demand.csv'
location_path = '../../UAI_data/input/location_cls.csv'
holiday_path = '../../UAI_data/input/holiday.csv'
weather_path = '../../UAI_data/input/weather.csv'
test_path = '../../UAI_data/input/test.csv'

def score_lgb(pred,valid):
    pred_l = pred
    valid_l = valid.get_label()
    sum_ = 0
    for i in range(len(valid_l)):
        sum_ += abs(pred_l[i] - valid_l[i])
    print sum_ / (1.0 * len(valid_l))
    return sum_ / (1.0 * len(valid_l))

def score(pred,valid):
    pred_l = list(pred)
    valid_l = list(valid)
    sum_ = 0
    for i in range(len(pred_l)):
        sum_ += abs(pred_l[i] - valid_l[i])
    return sum_ / (1.0 * len(pred_l))

def gbm_cv(x_train,y_train,params):
    # params = {
    #         'boosting_type': 'gbdt',
    #         'objective': 'regression',
    #         'min_child_weight':10,
    #         'metric': 'rmse',
    #         # 'num_leaves': 8,
    #         'num_leaves': 4,
    #         # 'learning_rate': 0.1,
    #         'learning_rate': 0.05,
    #         'feature_fraction': 0.8,
    #         'bagging_fraction': 0.8,
    #         'bagging_freq': 5,
    #         'verbose': 1,
    #         'lambda_l2': 1
    #     }

    train_data = lgb.Dataset(x_train, y_train)
    bst=lgb.cv(params,train_data, num_boost_round=5000, nfold=5, early_stopping_rounds=100)
    # print bst
    return bst

def training_offline(train,test):
    train_feats,test_feats = get_feats(train,test)
    print np.setdiff1d(train_feats.columns,test_feats.columns)
    print np.setdiff1d(test_feats.columns,train_feats.columns)
    do_not_use_list = ['create_date','demand_count','estimate_distance_mean','estimate_money_mean','estimate_term_mean','test_id','demand_count_start_h_rate','demand_count_weekday_max']
    predictors = [f for f in train_feats.columns if f not in do_not_use_list]
    print predictors
    # train_feats = train_feats[train_feats['create_date'] >= '2017-07-01'].copy()

    # import xgboost as xgb
    # params = {'min_child_weight': 10, 'eta': 0.05, 'colsample_bytree': 0.8, 'max_depth': 8,
    #                 'subsample': 0.8, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #                 'eval_metric': 'rmse', 'objective': 'reg:linear','seed':2017}
    # boostRound = 500
    #
    #
    # xgbtrain = xgb.DMatrix(train_feats_trip[predictors], train_feats_trip['demand_count'],missing=np.nan)
    # xgbvalid = xgb.DMatrix(test_feats[predictors],missing=np.nan)
    # model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
    # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # print param_score
    # test_feats.loc[:,'result'] = model.predict(xgbvalid)
    # test_feats['result'].fillna(1,inplace=True)

    params_lgb = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'min_child_weight':80,
            'metric': 'rmse',
            'num_leaves': 4,
            # 'num_leaves': 4,
            # 'learning_rate': 0.1,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
            'lambda_l2': 1,
            # 'feval':score,
            'seed':2017
        }
    X = train_feats[predictors]
    y = train_feats['demand_count']
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    train_data = lgb.Dataset(x_train,label=y_train,feature_name=predictors)

    bst=lgb.cv(params_lgb,train_data,num_boost_round=1000, nfold=5, early_stopping_rounds=30)
    print 'number of boosted round:',len(bst['rmse-mean'])
    lgb_model = lgb.train(params_lgb,train_data,num_boost_round=len(bst['rmse-mean']))

    # lgb_train = lgb.Dataset(x_train, y_train)
    # lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    # lgb_model = lgb.train(params_lgb,
    #                     lgb_train,
    #                     num_boost_round=2000,
    #                     feval=score_lgb,
    #                     valid_sets=lgb_eval,
    #                     early_stopping_rounds=100)

    print('Feature names:', lgb_model.feature_name())
    print('Feature importances:', list(lgb_model.feature_importance()))
    se = pd.Series(list(lgb_model.feature_importance()), index=lgb_model.feature_name())
    se.to_csv('param_offline.csv')
    print se
    test_feats.loc[:,'result'] = lgb_model.predict(test_feats[predictors])


    print 'predictive score is: ',score(test_feats['result'],test_feats['demand_count'])

def training_online(train,test):
    train_feats,test_feats = get_feats(train,test)
    print np.setdiff1d(train_feats.columns,test_feats.columns)
    print np.setdiff1d(test_feats.columns,train_feats.columns)
    # do_not_use_list = ['create_date','demand_count','estimate_distance_mean','estimate_money_mean','estimate_term_mean','test_id']
    do_not_use_list = ['create_date','demand_count','estimate_distance_mean','estimate_money_mean','estimate_term_mean','test_id','demand_count_start_h_rate']
    predictors = [f for f in train_feats.columns if f not in do_not_use_list]
    print predictors

    # train_feats = train_feats[train_feats['create_date'] >= '2017-07-22'].copy()
    # import xgboost as xgb
    # params = {'min_child_weight': 100, 'eta': 0.05, 'colsample_bytree': 0.8, 'max_depth': 8,
    #                 'subsample': 0.8, 'lambda': 1, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
    #                 'eval_metric': 'rmse', 'objective': 'reg:linear','seed':2017}
    # boostRound = 200
    #
    #
    # xgbtrain = xgb.DMatrix(train_feats_trip[predictors], train_feats_trip['demand_count'],missing=np.nan)
    # xgbtest = xgb.DMatrix(test_feats[predictors],missing=np.nan)
    # model = xgb.train(params, xgbtrain, num_boost_round=boostRound)
    # param_score = pd.Series(model.get_fscore()).sort_values(ascending=False)
    # print param_score
    #
    # test.loc[:,'count'] = model.predict(xgbtest)
    # test['count'].fillna(1,inplace=True)
    # test['test_id'] = test['test_id'].astype('int')
    # test[['test_id','count']].to_csv('result.csv',index=False)

    params_lgb = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'min_child_weight':80,
            'metric': 'rmse',
            'num_leaves': 4,
            # 'num_leaves': 4,
            # 'learning_rate': 0.1,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
            'lambda_l2': 1,
            'seed':2017
        }
    X = train_feats[predictors]
    y = train_feats['demand_count']
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    train_data = lgb.Dataset(x_train,label=y_train,feature_name=predictors)
    bst=lgb.cv(params_lgb,train_data, num_boost_round=1000, nfold=5, early_stopping_rounds=30)
    print 'number of boosted round:',len(bst['rmse-mean'])
    lgb_model = lgb.train(params_lgb,train_data,num_boost_round=len(bst['rmse-mean']))
    print('Feature names:', lgb_model.feature_name())
    print('Feature importances:', list(lgb_model.feature_importance()))
    se = pd.Series(list(lgb_model.feature_importance()), index=lgb_model.feature_name())
    se.to_csv('param_online.csv')
    print se

    test.loc[:,'count'] = lgb_model.predict(test_feats[predictors])
    test['count'].fillna(1,inplace=True)
    test['test_id'] = test['test_id'].astype('int')
    test[['test_id','count']].to_csv('result.csv',index=False)


def exclude_abnormal_value(train):
    coord = train.groupby(['create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'mean':'mean'})
    train = pd.merge(train, coord, on=['create_hour','start_geo_id','end_geo_id'],how='left')
    coord = train.groupby(['create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'std':'std'})
    train = pd.merge(train, coord, on=['create_hour','start_geo_id','end_geo_id'],how='left')
    train['mean'].fillna(method='bfill',inplace=True)
    train['std'].fillna(method='bfill',inplace=True)
    train.loc[:,'demand_count_min'] = train['mean'] - 2 * train['std']
    train.loc[:,'demand_count_max'] = train['mean'] + 2 * train['std']
    train['demand_count_max'] = np.ceil(train['demand_count_max'])
    max_bool = train['demand_count_max'] < train['demand_count']
    train['demand_count'][max_bool] = train['demand_count_max'][max_bool]
    neg_bool = train['demand_count_min'] < 0
    train['demand_count_min'][neg_bool] = 0
    min_bool = train['demand_count_min'] > train['demand_count']
    train['demand_count'][min_bool] = train['demand_count_min'][min_bool]
    train['demand_count'] = np.ceil(train['demand_count'])
    del train['demand_count_min'],train['demand_count_max'],train['std'],train['mean']
    return train


if __name__ == '__main__':
    t0 = time.time()
    # poi = pd.read_csv(poi_path,encoding='gbk')
    train_aug = pd.read_csv(train_Aug_path,encoding='gbk')
    train_jul = pd.read_csv(train_Jul_path,encoding='gbk')
    # train_jul_demand = pd.read_csv(train_jul_demand_path,encoding='gbk')
    # train_aug_demand = pd.read_csv(train_aug_demand_path,encoding='gbk')
    # loc_cls = pd.read_csv(location_path)
    # holiday = pd.read_csv(holiday_path)
    weather = pd.read_csv(weather_path)
    test = pd.read_csv(test_path,encoding='gbk')

    train = reshape_train(train_jul)

    # # Offline
    # train_tr = train[train['create_date'] >= '2017-07-01']
    # train_tr = train[train['create_date'] < '2017-07-25']
    # train_val = train[train['create_date'] >= '2017-07-25']
    # # print train_val.shape
    # train_tr = exclude_abnormal_value(train_tr)
    # valid = valid_split(train_val)
    # training_offline(train_tr,valid)

    # Online
    train = train[train['create_date'] >= '2017-07-01']
    train = exclude_abnormal_value(train)
    training_online(train,test)
    print(u'一共用时{}秒'.format(time.time()-t0))


