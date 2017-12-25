# -*- coding:utf-8 -*-
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table like and matrices
import pandas as pd
import numpy as np

# system
import os

cache_path = '../../UAI_data/output/cache/'
location_path = '../../UAI_data/input/location_cls.csv'
holiday_path = '../../UAI_data/input/holiday.csv'

def reshape_train(train):
    coord_jul = train.groupby(['create_date','create_hour','start_geo_id','end_geo_id'],as_index=False)['start_geo_id'].agg({'demand_count':'count'})
    coord_jul.loc[:,'estimate_money_mean'] = train.groupby(['create_date','create_hour','start_geo_id','end_geo_id'],as_index=False)['estimate_money'].mean()['estimate_money']
    coord_jul.loc[:,'estimate_distance_mean'] = train.groupby(['create_date','create_hour','start_geo_id','end_geo_id'],as_index=False)['estimate_distance'].mean()['estimate_distance']
    coord_jul.loc[:,'estimate_term_mean'] = train.groupby(['create_date','create_hour','start_geo_id','end_geo_id'],as_index=False)['estimate_term'].mean()['estimate_term']
    return coord_jul

def valid_split(train_valid,Shuffle=False):
    result_path = cache_path + 'valid_%d.hdf' %(train_valid.shape[0])
    if os.path.exists(result_path) & (Shuffle == False):
        valid = pd.read_hdf(result_path, 'w')
    else:
        train_valid.loc[:,'date_odd'] = train_valid['create_date'].map(lambda x: int(str(x).split('-')[2]) % 2)
        train_valid.loc[:,'hour_odd'] = train_valid['create_hour'].map(lambda x: int(x) % 2)
        # print train_val['date_odd'].value_counts()
        # print train_val['hour_odd'].value_counts()
        print train_valid.shape
        train_odd = train_valid[((train_valid['date_odd'] == 1) & (train_valid['hour_odd'] == 1))]
        train_even = train_valid[((train_valid['date_odd'] == 0) & (train_valid['hour_odd'] == 0))]
        train_valid = pd.concat([train_odd,train_even],axis=0)
        del train_valid['date_odd'],train_valid['hour_odd']
        coord_aug = train_valid.groupby(['create_date','create_hour','start_geo_id','end_geo_id'],as_index=False)['start_geo_id'].agg({'demand_count':'count'})
        # coord_aug = train_valid.groupby(['create_hour'],as_index=False)['create_hour'].agg({'demand_count':'count'})
        dict_list = {
            'l_50' : [0,1,2,3,4,5],
            'l_100' : [6],
            'l_150' : [12,23],
            'l_200' : [10,14,19],
            'l_250' : [7,8,11,15,16,18],
            'l_300' : [9,13],
            'l_350' : [17],
            'l_400' : [20,21]
        }
        l_22 = coord_aug[coord_aug['create_hour'] == 22].index
        r = np.random.choice(l_22,450)
        valid = coord_aug.iloc[r]
        for key,l in dict_list.items():
            num =  int(key.split('_')[1])
            for hour in l:
                l_tmp = coord_aug[coord_aug['create_hour'] == hour].index
                r = np.random.choice(l_tmp,num)
                tmp_valid = coord_aug.iloc[r].copy()
                valid = pd.concat([valid,tmp_valid],axis=0)
        valid.sort_values(['create_date','create_hour'],inplace=True)
        valid.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        # valid = train_valid
    return valid

def get_hol_feats(train,test):
    holiday = pd.read_csv(holiday_path)
    holiday['create_date'] = pd.to_datetime(holiday['create_date'])
    train['create_date'] = pd.to_datetime(train['create_date'])
    test['create_date'] = pd.to_datetime(test['create_date'])
    train = pd.merge(train,holiday,on='create_date',how='left')
    test = pd.merge(test,holiday,on='create_date',how='left')
    train['dayOfWeek'] = train['create_date'].dt.dayofweek
    test['dayOfWeek'] = test['create_date'].dt.dayofweek
    return train, test

def get_origin_feats(train,test):
    coord_money_hour = train.groupby(['create_hour','start_geo_id','end_geo_id'],as_index=False)['estimate_money_mean'].agg({'money_hour_mean':'mean'})
    coord_money_day = train.groupby(['dayOfWeek','start_geo_id','end_geo_id'],as_index=False)['estimate_money_mean'].agg({'money_day_mean':'mean'})
    coord_dis_hour = train.groupby(['start_geo_id','end_geo_id'],as_index=False)['estimate_distance_mean'].agg({'dis_hour_mean':'mean'})
    coord_term_hour = train.groupby(['start_geo_id','end_geo_id'],as_index=False)['estimate_term_mean'].agg({'term_hour_mean':'mean'})
    train = pd.merge(train,coord_money_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    train = pd.merge(train,coord_money_day,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    train = pd.merge(train,coord_dis_hour,on=['start_geo_id','end_geo_id'],how='left')
    train = pd.merge(train,coord_term_hour,on=['start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_money_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_money_day,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_dis_hour,on=['start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_term_hour,on=['start_geo_id','end_geo_id'],how='left')
    return train,test

def get_loc_cluster_feats(train,test):
    loc_cls = pd.read_csv(location_path)
    loc_start = loc_cls.copy()
    loc_start.rename(columns={'location_id':'start_geo_id','cluster':'start_cluster'},inplace=True)
    train = pd.merge(train,loc_start,on='start_geo_id',how='left')
    loc_end = loc_cls.copy()
    loc_end.rename(columns={'location_id':'end_geo_id','cluster':'end_cluster'},inplace=True)
    train = pd.merge(train,loc_end,on='end_geo_id',how='left')
    loc_start = loc_cls.copy()
    loc_start.rename(columns={'location_id':'start_geo_id','cluster':'start_cluster'},inplace=True)
    test = pd.merge(test,loc_start,on='start_geo_id',how='left')
    loc_end = loc_cls.copy()
    loc_end.rename(columns={'location_id':'end_geo_id','cluster':'end_cluster'},inplace=True)
    test = pd.merge(test,loc_end,on='end_geo_id',how='left')
    return train,test

def get_loc_feats(train,test):
    coord_se_mean = train.groupby(['start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'sloc_eloc_mean':'mean'})
    coord_se_median = train.groupby(['start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'sloc_eloc_median':'median'})
    coord_se_max = train.groupby(['start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'sloc_eloc_max':'max'})
    coord_se_min = train.groupby(['start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'sloc_eloc_min':'min'})
    train = pd.merge(train,coord_se_mean,on=['start_geo_id','end_geo_id'],how='left')
    train = pd.merge(train,coord_se_median,on=['start_geo_id','end_geo_id'],how='left')
    train = pd.merge(train,coord_se_max,on=['start_geo_id','end_geo_id'],how='left')
    train = pd.merge(train,coord_se_min,on=['start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_se_mean,on=['start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_se_median,on=['start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_se_max,on=['start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_se_min,on=['start_geo_id','end_geo_id'],how='left')

    coord_se_mean = train.groupby(['start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'sloccl_eloccl_mean':'mean'})
    coord_se_median = train.groupby(['start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'sloccl_eloccl_median':'median'})
    coord_se_max = train.groupby(['start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'sloccl_eloccl_max':'max'})
    coord_se_min = train.groupby(['start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'sloccl_eloccl_min':'min'})
    train = pd.merge(train,coord_se_mean,on=['start_cluster','end_cluster'],how='left')
    train = pd.merge(train,coord_se_median,on=['start_cluster','end_cluster'],how='left')
    train = pd.merge(train,coord_se_max,on=['start_cluster','end_cluster'],how='left')
    train = pd.merge(train,coord_se_min,on=['start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_se_mean,on=['start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_se_median,on=['start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_se_max,on=['start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_se_min,on=['start_cluster','end_cluster'],how='left')

    coord_se_sum = train.groupby(['start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'sloc_eloc_sum':'sum'})
    coord_s_sum = coord_se_sum.groupby('start_geo_id',as_index=False)['sloc_eloc_sum'].agg({'sloc_sum':'sum'})
    coord_e_sum = coord_se_sum.groupby('end_geo_id',as_index=False)['sloc_eloc_sum'].agg({'eloc_sum':'sum'})
    coord_se_sum = pd.merge(coord_se_sum,coord_s_sum,on='start_geo_id',how='left')
    coord_se_sum = pd.merge(coord_se_sum,coord_e_sum,on='end_geo_id',how='left')
    coord_se_sum.loc[:,'se_start_ratio'] = coord_se_sum['sloc_eloc_sum'] / (1.0 * coord_se_sum['sloc_sum'])
    coord_se_sum.loc[:,'se_end_ratio'] = coord_se_sum['sloc_eloc_sum'] / (1.0 * coord_se_sum['eloc_sum'])
    del coord_se_sum['sloc_eloc_sum'],coord_se_sum['sloc_sum'],coord_se_sum['eloc_sum']
    train = pd.merge(train,coord_se_sum,on=['start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_se_sum,on=['start_geo_id','end_geo_id'],how='left')

    coord_se_sum = train.groupby(['start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'sloc_eloc_sum':'sum'})
    coord_s_sum = coord_se_sum.groupby('start_cluster',as_index=False)['sloc_eloc_sum'].agg({'sloc_sum':'sum'})
    coord_e_sum = coord_se_sum.groupby('end_cluster',as_index=False)['sloc_eloc_sum'].agg({'eloc_sum':'sum'})
    coord_se_sum = pd.merge(coord_se_sum,coord_s_sum,on='start_cluster',how='left')
    coord_se_sum = pd.merge(coord_se_sum,coord_e_sum,on='end_cluster',how='left')
    coord_se_sum.loc[:,'se_start_cls_ratio'] = coord_se_sum['sloc_eloc_sum'] / (1.0 * coord_se_sum['sloc_sum'])
    coord_se_sum.loc[:,'se_end_cls_ratio'] = coord_se_sum['sloc_eloc_sum'] / (1.0 * coord_se_sum['eloc_sum'])
    del coord_se_sum['sloc_eloc_sum'],coord_se_sum['sloc_sum'],coord_se_sum['eloc_sum']
    train = pd.merge(train,coord_se_sum,on=['start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_se_sum,on=['start_cluster','end_cluster'],how='left')

    return train,test

def time_handler(train,test):
    map_hour = {1:1,2:1,3:1,4:1,5:1,6:1,7:2,8:2,9:2,10:3,11:3,12:3,13:4,14:4,15:4,16:5,17:5,18:5,19:6,20:6,21:6,22:0,23:0,0:0}
    train.loc[:,'hour_cls'] = train['create_hour'].map(lambda x: map_hour[x])
    test.loc[:,'hour_cls'] = test['create_hour'].map(lambda x: map_hour[x])
    return train,test

def get_hour_demand(train,test):
    coord_hour = train.groupby(['create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_h_avg':'mean'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_h_median':'median'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_h_max':'max'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_h_std':'std'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_geo_id','end_geo_id'],how='left')

    coord_hour_s = train.groupby(['create_hour','start_geo_id'],as_index=False)['demand_count'].agg({'demand_count_same_start_h_avg':'mean'})
    coord_hour_e = train.groupby(['create_hour','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_same_end_h_avg':'mean'})
    train = pd.merge(train,coord_hour_s,on=['create_hour','start_geo_id'],how='left')
    test = pd.merge(test,coord_hour_s,on=['create_hour','start_geo_id'],how='left')
    train = pd.merge(train,coord_hour_e,on=['create_hour','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour_e,on=['create_hour','end_geo_id'],how='left')

    train.loc[:,'demand_count_start_h_rate'] = train['demand_count_h_avg'] / (train['demand_count_same_start_h_avg'] + 0.01)
    test.loc[:,'demand_count_start_h_rate'] = test['demand_count_h_avg'] / (test['demand_count_same_start_h_avg'] + 0.01)
    train.loc[:,'demand_count_end_h_rate'] = train['demand_count_h_avg'] / (train['demand_count_same_end_h_avg'] + 0.01)
    test.loc[:,'demand_count_end_h_rate'] = test['demand_count_h_avg'] / (test['demand_count_same_end_h_avg'] + 0.01)
    del train['demand_count_same_end_h_avg'],train['demand_count_same_start_h_avg']
    del test['demand_count_same_end_h_avg'],test['demand_count_same_start_h_avg']

    return train, test

def get_weekday_demand(train,test):
    coord_hour = train.groupby(['dayOfWeek','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_weekday_avg':'mean'})
    train = pd.merge(train,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['dayOfWeek','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_weekday_median':'median'})
    train = pd.merge(train,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['dayOfWeek','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_weekday_max':'max'})
    train = pd.merge(train,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['dayOfWeek','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_weekday_std':'std'})
    train = pd.merge(train,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['dayOfWeek','start_geo_id','end_geo_id'],how='left')

    return train, test

def get_hour_loc_cls_demand(train,test):
    coord_hour = train.groupby(['create_hour','start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'demand_count_cls_h_avg':'mean'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')
    coord_hour = train.groupby(['create_hour','start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'demand_count_cls_h_median':'median'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')
    coord_hour = train.groupby(['create_hour','start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'demand_count_cls_h_max':'max'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')
    coord_hour = train.groupby(['create_hour','start_cluster','end_cluster'],as_index=False)['demand_count'].agg({'demand_count_cls_h_std':'std'})
    train = pd.merge(train,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')
    test = pd.merge(test,coord_hour,on=['create_hour','start_cluster','end_cluster'],how='left')

    coord_hour_s = train.groupby(['create_hour','start_cluster'],as_index=False)['demand_count'].agg({'demand_count_same_start_cls_h_avg':'mean'})
    coord_hour_e = train.groupby(['create_hour','end_cluster'],as_index=False)['demand_count'].agg({'demand_count_same_end_cls_h_avg':'mean'})
    train = pd.merge(train,coord_hour_s,on=['create_hour','start_cluster'],how='left')
    test = pd.merge(test,coord_hour_s,on=['create_hour','start_cluster'],how='left')
    train = pd.merge(train,coord_hour_e,on=['create_hour','end_cluster'],how='left')
    test = pd.merge(test,coord_hour_e,on=['create_hour','end_cluster'],how='left')

    train.loc[:,'demand_count_start_h_rate'] = train['demand_count_h_avg'] / (train['demand_count_same_start_cls_h_avg'] + 0.01)
    test.loc[:,'demand_count_start_h_rate'] = test['demand_count_h_avg'] / (test['demand_count_same_start_cls_h_avg'] + 0.01)
    train.loc[:,'demand_count_end_h_rate'] = train['demand_count_h_avg'] / (train['demand_count_same_end_cls_h_avg'] + 0.01)
    test.loc[:,'demand_count_end_h_rate'] = test['demand_count_h_avg'] / (test['demand_count_same_end_cls_h_avg'] + 0.01)
    del train['demand_count_same_end_cls_h_avg'],train['demand_count_same_start_cls_h_avg']
    del test['demand_count_same_end_cls_h_avg'],test['demand_count_same_start_cls_h_avg']

    return train, test

def get_hour_cls_demand(train,test):
    coord_hour = train.groupby(['hour_cls','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_avg':'mean'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['hour_cls','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_median':'median'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['hour_cls','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_max':'max'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['hour_cls','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_std':'std'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')

    return train, test

def get_hol_demand(train,test):
    coord_hour = train.groupby(['holiday','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_avg':'mean'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['hour_cls','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_median':'median'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['hour_cls','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_max':'max'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['hour_cls','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hc_std':'std'})
    train = pd.merge(train,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['hour_cls','start_geo_id','end_geo_id'],how='left')

    return train, test

def get_hol_demand(train,test):
    coord_hour = train.groupby(['holiday','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_avg':'mean'})
    train = pd.merge(train,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['holiday','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_median':'median'})
    train = pd.merge(train,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['holiday','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_max':'max'})
    train = pd.merge(train,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['holiday','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_std':'std'})
    train = pd.merge(train,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','start_geo_id','end_geo_id'],how='left')

    return train, test

def get_hol_h_demand(train,test):
    coord_hour = train.groupby(['holiday','create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_h_avg':'mean'})
    train = pd.merge(train,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['holiday','create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_h_median':'median'})
    train = pd.merge(train,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['holiday','create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_h_max':'max'})
    train = pd.merge(train,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')
    coord_hour = train.groupby(['holiday','create_hour','start_geo_id','end_geo_id'],as_index=False)['demand_count'].agg({'demand_count_hol_h_std':'std'})
    train = pd.merge(train,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')
    test = pd.merge(test,coord_hour,on=['holiday','create_hour','start_geo_id','end_geo_id'],how='left')

    return train, test


def get_feats(train,test):
    train,test = get_hol_feats(train,test)
    train,test = get_origin_feats(train,test)
    train,test = get_loc_cluster_feats(train,test)
    train,test = get_loc_feats(train,test)
    train,test = time_handler(train,test)
    train,test = get_hour_demand(train,test)
    train,test = get_hol_demand(train,test)
    # train,test = get_hol_h_demand(train,test)
    train,test = get_weekday_demand(train,test)
    train,test = get_hour_loc_cls_demand(train,test)
    train,test = get_hour_cls_demand(train,test)
    return train,test