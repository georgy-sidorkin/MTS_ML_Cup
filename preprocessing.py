"""
Аггрегация данных. Версия 3
"""

import os
import pandas as pd
import numpy as np
import datetime
import tqdm
import warnings

warnings.filterwarnings('ignore')


def get_data_part_day(data: pd.DataFrame) -> pd.DataFrame:
    df_part_day = pd.get_dummies(data[['user_id', 'part_of_day']])

    df_part_day = (df_part_day.groupby('user_id', as_index=False)
                   .agg({'part_of_day_day': 'sum',
                         'part_of_day_evening': 'sum',
                         'part_of_day_morning': 'sum',
                         'part_of_day_night': 'sum'}))

    df_part_day['sum_visits'] = (df_part_day.part_of_day_day
                                 + df_part_day.part_of_day_evening
                                 + df_part_day.part_of_day_morning
                                 + df_part_day.part_of_day_night)

    df_part_day['day_pct'] = df_part_day.part_of_day_day / df_part_day.sum_visits
    df_part_day['evening_pct'] = df_part_day.part_of_day_evening / df_part_day.sum_visits
    df_part_day['morning_pct'] = df_part_day.part_of_day_morning / df_part_day.sum_visits
    df_part_day['night_pct'] = df_part_day.part_of_day_night / df_part_day.sum_visits
    return df_part_day


def get_data_days(data: pd.DataFrame) -> pd.DataFrame:
    df_active_days = (data.groupby('user_id', as_index=False)
                      .agg({'date': 'nunique',
                            'request_cnt': 'sum'})
                      .rename(columns={'date': 'act_days'}))

    df_active_days['avg_req_per_day'] = (df_active_days.request_cnt /
                                         df_active_days.act_days)

    df_req_std = (data.groupby('user_id', as_index=False)
                  .agg({'request_cnt': np.std})
                  .rename(columns={'request_cnt': 'requests_std'})
                  .fillna(0))

    df_dates_period = (data.groupby('user_id', as_index=False)
                       .agg({'date': ['max', 'min']}))

    df_dates_period['period_days'] = (df_dates_period['date'].iloc[:, 0] -
                                      df_dates_period['date'].iloc[:, 1])
    df_dates_period['period_days'] = df_dates_period.period_days.dt.days + 1
    df_dates_period = df_dates_period.drop('date', axis=1)

    df_dates_period.columns = df_dates_period.columns.droplevel(1)

    df_days = (df_active_days
               .merge(df_dates_period, on='user_id')
               .merge(df_req_std, on='user_id')
               )

    df_days['act_days_pct'] = df_days.act_days / df_days.period_days

    df_days['request_cnt_act_days'] = df_days.act_days_pct * df_days.request_cnt
    return df_days


def get_user_model_price(data: pd.DataFrame) -> pd.DataFrame:
    data.cpe_model_os_type.replace('Apple iOS', 'iOS', inplace=True)

    data.cpe_manufacturer_name.replace('Huawei Device Company Limited', 'Huawei', inplace=True)
    data.cpe_manufacturer_name.replace('Motorola Mobility LLC, a Lenovo Company', 'Motorola', inplace=True)
    data.cpe_manufacturer_name.replace('Sony Mobile Communications Inc.', 'Sony', inplace=True)

    data['cpe_type_cd'] = np.where((data['cpe_manufacturer_name'] == 'Nokia') & (data['cpe_model_name'] == '3 Dual'),
                                   'plain', data['cpe_type_cd'])
    df_model = data.groupby(['user_id',
                             'cpe_type_cd',
                             'cpe_model_os_type',
                             'cpe_manufacturer_name'], as_index=False).agg({'price': 'mean'})
    return df_model.fillna(-999)


def get_user_city_cnt(data: pd.DataFrame) -> pd.DataFrame:
    df_city_cnt = data.groupby('user_id', as_index=False) \
        .agg({'region_name': 'nunique',
              'city_name': 'nunique'}) \
        .rename(columns={'region_name': 'region_cnt',
                         'city_name': 'city_cnt'})
    return df_city_cnt


def get_user_url_cnt(data: pd.DataFrame) -> pd.DataFrame:
    df_url_cnt = data.groupby('user_id', as_index=False) \
        .agg({'url_host': 'nunique',
              'domain': 'nunique'}) \
        .rename(columns={'url_host': 'url_host_cnt',
                         'domain': 'domain_cnt'})
    return df_url_cnt


DATA_FILE = 'competition_data_final_pqt'


if __name__ == "__main__":
    for file in tqdm.tqdm(os.listdir(DATA_FILE)):
        if file.endswith('.parquet'):
            df = pd.read_parquet(f'{DATA_FILE}/{file}', engine='pyarrow')

            df['domain'] = df.url_host.transform(lambda x: x.split('.')[-1])

            data_part_day = get_data_part_day(df)
            data_days = get_data_days(df)
            data_user_model = get_user_model_price(df)
            data_city_cnt = get_user_city_cnt(df)
            data_url_cnt = get_user_url_cnt(df)

            df_final = data_part_day.merge(data_days, how='left', on='user_id') \
                .merge(data_user_model, how='left', on='user_id') \
                .merge(data_city_cnt, how='left', on='user_id') \
                .merge(data_url_cnt, how='left', on='user_id')

            df_final.name = file[:10]

            out_name = f'data_agg_{df_final.name}'

            d = str(datetime.date.today())
            out_dir = 'data_agg' + '/' + d
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            final_path = os.path.join(out_dir, out_name) + '.csv'

            df_final.to_csv(final_path, encoding='utf-8', index=False)
