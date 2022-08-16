# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import os
from pandas.tseries.offsets import YearEnd


# ## 1) Get and Check Data

# ### 1-1) WorldBank Open Data

def load_world_bank_data(filepath, col_name, countries = ['KOR','CAN']):
    df = pd.read_excel(filepath, sheet_name = 'Data', skiprows = 3)
    df = df.set_index('Country Code')
    tmp_df = []
    for country in countries:
        cntry_df = df.loc[country].to_frame()
        cntry_df = cntry_df.iloc[3:]
        cntry_df = cntry_df.dropna()
        cntry_df[country] = cntry_df[country].astype('float')
        cntry_df['date'] = pd.to_datetime(cntry_df.index) + YearEnd(0) # 자동으로 날짜별로 게산  + 연말일자 
        cntry_df = cntry_df.set_index('date')
        cntry_df.columns = [col_name + '_' + country]
        tmp_df.append(cntry_df)
    result_df = pd.concat(tmp_df, axis = 1)
    return result_df


df1 = load_world_bank_data('./data/wb/GDP.xls', 'GDP', ['KOR','CAN','USA'])
df2 = load_world_bank_data('./data/wb/GDP_growth.xls', 'GDP_GROWTH', ['KOR','CAN','USA'])
df3 = load_world_bank_data('./data/wb/PPP_XR.xls', 'PPP_XR', ['KOR','CAN','USA'])
df4 = load_world_bank_data('./data/wb/GDP_per_capita_growth.xls', 'GDP/CAPITA_GROWTH', ['KOR','CAN','USA'])

wb_df_dict = {'GDP':df1, 'GDP_GROWTH':df2, 'PPP_XR':df3, 'GDP/CAPITA_GROWTH': df4}

# > GDP, GDP증감, PPP_XR, 인구증가율 대비 GDP 4개 항목 입수 

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 8

plt.subplots(2,2, figsize = (10,6))
for i, k in enumerate(wb_df_dict.keys()):
    plt.subplot(2,2,i+1)
    label = [col.replace(k+'_','') for col in wb_df_dict[k].columns]
    plt.plot(wb_df_dict[k], label = label)
    plt.legend()
    plt.title(k)
plt.show()

# > 연도별 데이터로 보유 시점이 각각 다르므로 모든 기간 보유한 시점으로 확인 

wb_df = pd.merge(df1, df2, left_index = True, right_index = True, how = 'left')
wb_df = pd.merge(wb_df, df3, left_index = True, right_index = True, how = 'left')
wb_df = pd.merge(wb_df, df4, left_index = True, right_index = True, how = 'left')

import seaborn as sns

plt.figure(figsize = (8,6))
sns.heatmap(wb_df.isnull())

# > 대충 이런 결측친데 IMF & CAN GDP 시점으로 99년도 정도부터 쓰면될것 처럼 보임  

wb_df.dropna(inplace = True)

wb_df.head(3)

# > GDP 항목은 직접 사용하진 않지만 growth 검증을 위해 냅두기로..  

wb_df.to_excel('./data/full_wb.xlsx')

os.listdir('./data/ds')


