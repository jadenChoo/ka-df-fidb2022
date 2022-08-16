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

import os
from pandas.tseries.offsets import YearEnd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# # 1. 데이터 입수   
# ---

# ## 1-1) World Bank Data  

# - ppp xr  
# - gdp constant 
# - gdp current
# - gdp growth  

def load_world_bank_data(filepath, col_name, countries = ['KOR','CAN','USA']):
    df = pd.read_excel(filepath, sheet_name = 'Data', skiprows = 3)
    df = df.set_index('Country Code')
    dfs = []
    for country in countries:
        cntry_df = df.loc[country].to_frame()
        cntry_df = cntry_df.iloc[3:]
        cntry_df = cntry_df.dropna()
        cntry_df[country] = cntry_df[country].astype('float')
        cntry_df['date'] = pd.to_datetime(cntry_df.index) + YearEnd(0) # 자동으로 날짜별로 게산  + 연말일자 
        cntry_df = cntry_df.set_index('date')
        cntry_df.columns = [col_name + '_' + country]
        dfs.append(cntry_df)
    dfs = pd.concat(dfs, axis = 1)
    return dfs


filepaths = f'./data/wb/'
wb_dfs = []
files = os.listdir(filepaths)
for file in files:
    wb_df = load_world_bank_data(filepaths + file, file.split('.')[0])
    wb_dfs.append(wb_df)
wb_dfs = pd.concat(wb_dfs,axis = 1)

wb_dfs.head(3)


# > World Bank 데이터는 Annual 데이터만 제공  
# > 실질 GDP 단위가 이상합니다 

# ## 1-2) Deep Search Data

# - GDP
# - 경제성장율
#   
#   
# - 경상수지
# - 국채1년물
# - 기준금리
# - PPI
# - CPI
# - 실업률
# - 외환보유액
# - 환율

def load_deepsearch_data(filepath, col_name):
    df = pd.read_excel(filepath)
    df.columns = ['date', col_name]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


filepath = f'./data/ds/'
ds_dfs = []
for file in os.listdir(filepath):
    ds_df = load_deepsearch_data(filepath + file, file.split('.')[0])
    ds_dfs.append(ds_df)
ds_dfs = pd.concat(ds_dfs, axis = 1)    

# #### Check) WB/DS Data Validation  
# - 딥서치 제공 gdp 성격 확인  

print('Deep Search gdp      : {}'.format(ds_dfs.loc['2020-12-31','GDP_한국']))
print('World Bank  명목 gdp : {}'.format(wb_dfs.loc['2020-12-31','GDP_CURRENT_KOR']))
print('World Bank  실질 gdp : {}'.format(wb_dfs.loc['2020-12-31','GDP_CONSTANT_KOR']))

# > 딥서치 GDP 정보는 명목 총생산금액임.

# * 경제성장율은?  
# - world bank gdp growth 는 실질 gdp 기준으로 산출  

ds_dfs.loc['2021-01-31':'2021-12-31','경제성장율_한국'].dropna()

wb_dfs[['GDP_GROWTH_KOR']].dropna().tail(5)

# > 딥서치 경제성장율은 실질gdp의 성장율, 단 분기별 성장율이 집계 되어 있음  

# ## 1-3) Merge and Imputation

full_df = pd.concat([ds_dfs, wb_dfs], axis = 1)

full_df.head(3)

# 월말 데이터만 남겨두고 
full_df = full_df.resample('M').last()

full_df = full_df.fillna(method = 'ffill')

plt.figure(figsize = (12,4))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(full_df.loc['2000-01-31':].isnull(), cmap=cmap, vmax=1, vmin = 0, center=0,
#              linewidths=.5, 
            cbar_kws={"shrink": .5})
plt.yticks([])
plt.show()

# > 한국의 외환보유액을 사용할 수 있는 시점으로 데이터 수집 시점을 잡고, 중국의 외환보유액은 항목에서 삭제  

full_df.drop('외환보유액_중국', axis = 1, inplace = True)

full_df = full_df[full_df['외환보유액_한국'].notnull()]

# ## 2. 데이터 전처리 
# - 변수별 단위가 상이한 것으로 보임.  
# - 메타 관리차원에서 한번 항목별 체크하고 진행 

full_df.columns

# ### 2-1) GDP 정비  
# - 딥서치랑 월드뱅크에서 가져온 데이터 체크 
#
#
#     # 분기단위 실질GDP 변수 추가 
#     # 캐나다 실질GDP or 경제성장율 숫자 싱크가 안맞음.

full_df[['GDP_미국','GDP_CONSTANT_USA','GDP_CURRENT_USA']].tail(3)

full_df['명목GDP_미국'] = full_df['GDP_미국']*1000000
full_df['명목GDP_한국'] = full_df['GDP_한국']*1000000
full_df['명목GDP_캐나다'] = full_df['GDP_캐나다']*1000000

# > 명목gdp는 딥서치 데이터를 활용하며 단위는 달러 단위로 통일  
# wb에서 가져온 명목 gdp는 삭제  

# > wb 경제성장율은 연간 정보니깐 제거  
# > 실질 gdp는 real xr 산출에서 사용될 것. 연 단위 데이터이므로 분기단위 성장율을 통해 분기단위 실질 gdp 계산해서 활용도 가능할듯

# #### 분기단위 실질 GDP 를 구할 수 있음(아래서 구해보자 )

# 분기 단위 정보, 실질 gdp기준으로 만들어져 있음  
full_df[['경제성장율_미국','경제성장율_캐나다','경제성장율_한국']]

# * 분기별 성장율을 sumproduct 하여 실제 실질성장율을 분기별로 계산 숫자의 오차는 어느 정도인지 체크 

# +
# 20년 말 기준 실질gdp로 성장율로 계산 시 21년 말 gdp가 나오는지?
t1 = full_df.loc['2020-12-31','GDP_CONSTANT_USA']*(100+full_df.loc['2021-03-31','경제성장율_미국'])/100\
*(100+full_df.loc['2021-06-30','경제성장율_미국'])/100\
*(100+full_df.loc['2021-09-30','경제성장율_미국'])/100\
*(100+full_df.loc['2021-12-31','경제성장율_미국'])/100
t2 = full_df.loc['2021-12-31','GDP_CONSTANT_USA']

t1/t2
# -

# > 99.8% 정도.. 아마 단위 절사 문제로 보임. 그래도 이게 나을듯 

# 반대로도 한번 체크 
t1 = full_df.loc['2021-12-31','GDP_CONSTANT_USA']*(100-full_df.loc['2021-12-31','경제성장율_미국'])/100\
*(100-full_df.loc['2021-09-30','경제성장율_미국'])/100\
*(100-full_df.loc['2021-06-30','경제성장율_미국'])/100\
*(100-full_df.loc['2021-03-31','경제성장율_미국'])/100
t2 = full_df.loc['2020-12-31','GDP_CONSTANT_USA']
t2/t1

# > 이쪽도 문제 없어보임.  분기별로 실질 gdp 재연산 진행 

sub_df = full_df[['GDP_CONSTANT_USA','GDP_CONSTANT_KOR','GDP_CONSTANT_CAN','경제성장율_미국','경제성장율_한국','경제성장율_캐나다']].resample('Q').last()

real_gdp = sub_df.iloc[0]['GDP_CONSTANT_USA']
for idx in sub_df.index:
    gap_rt = 1 + sub_df.loc[idx,'경제성장율_미국']/100 # 12월 발표였으니 그냥 바로 곱해도 가능 
    real_gdp *= gap_rt
    sub_df.loc[idx,'실질GDP_미국'] = real_gdp

sub_df.tail(3)

sub_df.loc['2021-12-31', 'GDP_CONSTANT_USA'] / sub_df.loc['2021-12-31', 'GDP_CONSTANT_USA']

# > 연말숫자 일치 확인 완료 차트로 실효성 확인 

# #### 미국 분기 실질GDP추정

    plt.plot(sub_df[['GDP_CONSTANT_USA','실질GDP_미국']])
    plt.show()

# > 등락이 큰 부분들이 추가로 반영되는 효과를 볼 수 있음.  
# 나머지 국가도 동일하게 작업  

# #### 한국 실질 GDP추정

# 한국 실질 gdp 단위가 좀 이상한 듯함..  맞추고 가자  
full_df.loc['2020-12-31','GDP_CONSTANT_KOR'] 

# > 1835 조 달러 라고 나오는데..  구글에서 찾아보면 1.6조 달러 수준. 1000단위로 한번 나눠줘야 할듯 

# 한국
real_gdp = sub_df.iloc[0]['GDP_CONSTANT_KOR']
for idx in sub_df.index:
    gap_rt = 1 + sub_df.loc[idx,'경제성장율_한국']/100 # 12월 발표였으니 그냥 바로 곱해도 가능 
    real_gdp *= gap_rt
    sub_df.loc[idx,'실질GDP_한국'] = real_gdp
plt.plot(sub_df[['GDP_CONSTANT_KOR','실질GDP_한국']])
plt.show()

sub_df['실질GDP_한국'] /= 1000

# #### 캐나다!?

# 캐나다    
real_gdp = sub_df.iloc[0]['GDP_CONSTANT_CAN']
for idx in sub_df.index:
    gap_rt = 1 + sub_df.loc[idx,'경제성장율_캐나다']/100 # 12월 발표였으니 그냥 바로 곱해도 가능 
    real_gdp *= gap_rt
    sub_df.loc[idx,'실질GDP_캐나다'] = real_gdp    
plt.plot(sub_df[['GDP_CONSTANT_CAN','실질GDP_캐나다']])
plt.show()

# > 갭이 좀 크게 발생함. 사유는?

sub_df['체크'] = sub_df['GDP_CONSTANT_CAN']/sub_df['실질GDP_캐나다']
sub_df[['GDP_CONSTANT_CAN','경제성장율_캐나다','실질GDP_캐나다','체크']].head(20)

# 로그 계산을 해보아도 

sub_df['log_growth'] = np.log(1 + sub_df['경제성장율_캐나다']/100) # percent convert
sub_df['rsum_log_growth'] = sub_df['log_growth'].rolling(4).sum()
sub_df['rprod'] = (np.exp(sub_df['rsum_log_growth']) - 1)*100
sub_df[['GDP_CONSTANT_CAN','rprod','경제성장율_캐나다','실질GDP_캐나다']]

# > 캐나다는 숫자가 안맞음..  
# 실제 계산해보아도 경제성장율 sumproduct 와 실질gdp값이 일치하지 않음.  
# 우선 계산된 값으로 사용하고 사유는 파악 필요 

sub_df = sub_df[['실질GDP_미국','실질GDP_캐나다','실질GDP_한국']]

full_df = pd.concat([full_df, sub_df], axis = 1)

# 분기정보 채웠으니 다시 ffill
full_df = full_df.fillna(method = 'ffill')

full_df.columns

# ### 2-2) 경상수지 관련  
# - 수업중에는 경상수지 / 명목GDP 를 지표로 활용하였음.  
# - 경상수지가 인플레이션 반영된 숫자이므로 동일하게 연산 
# - 각 정보는 백만달러 단위  
#
#

# # 경상수지/실질GDP(분기) 변수 추가 
# # 실질gdp단위환산 

full_df['경상수지_한국']

# > https://www.index.go.kr/potal/main/EachDtlPageDetail.do?idx_cd=2735 참고  
# 5월 한국 경상수지는 3,860 백 만불, 월 단위임

# 단위 변환해주고 
for c in ['경상수지_한국','경상수지_캐나다','경상수지_미국']:
    full_df['조정'+c] = full_df[c] * 1000000

plt.plot(full_df[['조정경상수지_한국','조정경상수지_캐나다','조정경상수지_미국']], label = ['조정경상수지_한국','조정경상수지_캐나다','조정경상수지_미국'])
plt.legend()
plt.show()

# 경상수지를 index화 시키고 
full_df['명목GDP대비경상수지_미국'] = full_df['조정경상수지_미국']/full_df['명목GDP_미국']
full_df['명목GDP대비경상수지_캐나다'] = full_df['조정경상수지_캐나다']/full_df['명목GDP_캐나다']
full_df['명목GDP대비경상수지_한국'] = full_df['조정경상수지_한국']/full_df['명목GDP_한국']

# +
# full_df.drop(['명목GDP_미국','명목GDP_캐나다','명목GDP_한국'], axis = 1, inplace = True)
# full_df.drop(['GDP_CONSTANT_KOR','GDP_CONSTANT_USA','GDP_CONSTANT_CAN'], axis = 1, inplace = True)
# full_df.drop(['GDP_GROWTH_CAN','GDP_GROWTH_KOR','GDP_GROWTH_USA'], axis = 1, inplace = True)
# full_df.drop(['GDP_미국','GDP_한국','GDP_캐나다','GDP_CURRENT_USA','GDP_CURRENT_CAN','GDP_CURRENT_KOR'], axis = 1,inplace = True)
# full_df.drop(['경상수지_미국','경상수지_캐나다','경상수지_한국'], axis = 1, inplace = True)
# -

# * 명목gdp = 실질gdp + 인플레이션 
# * 인플레이션 = 양국 금리차이 or 양국 cpi 차이  

# > 명목GDP는 연단위이므로 실질GDP를 활용해서 경상수지 인덱스를 만든다면 ?  
# 실질GDP + 인플레이션으로 나눠보고 싶지만 우선 진행

# 경상수지를 index화 시키고 
full_df['실질GDP대비경상수지_미국'] = full_df['조정경상수지_미국']/full_df['실질GDP_미국']
full_df['실질GDP대비경상수지_캐나다'] = full_df['조정경상수지_캐나다']/full_df['실질GDP_캐나다']
full_df['실질GDP대비경상수지_한국'] = full_df['조정경상수지_한국']/full_df['실질GDP_한국']

full_df[['실질GDP대비경상수지_미국','실질GDP대비경상수지_캐나다','실질GDP대비경상수지_한국']]

features = ['실질GDP대비경상수지_미국','명목GDP대비경상수지_미국']
plt.plot(full_df[features], label = features)
plt.legend()
plt.show()

features = ['실질GDP대비경상수지_캐나다','명목GDP대비경상수지_캐나다']
plt.plot(full_df[features], label = features)
plt.legend()
plt.show()

features = ['실질GDP대비경상수지_한국','명목GDP대비경상수지_한국']
plt.plot(full_df[features], label = features)
plt.legend()
plt.show()

# > 한국의 실질GDP대비경상수지는 왜? => 위에서 단위 맞춰서 해결 

# > 한국의 경상수지는 월별 데이터이지만 나머지는 분기별 데이터임.  
# 명목GDP는 연별 데이터  

full_df.columns

# ### 2-3) 경제성장율 
# - 실질경제성장율을 그냥 사용하면 될듯  

# ### 2-4) 인플레이션
# - 물가지수 활용 
# - 기준금리 활용

plt.plot(full_df[['소비자물가지수_한국','소비자물가지수_미국','소비자물가지수_캐나다']].tail(12))
plt.show()

# > 데이터는 월별로 존재

full_df['CPI인플레이션_한국'] = full_df['소비자물가지수_한국'].pct_change(1)
full_df['CPI인플레이션_미국'] = full_df['소비자물가지수_미국'].pct_change(1)
full_df['CPI인플레이션_캐나다'] = full_df['소비자물가지수_캐나다'].pct_change(1)

plt.plot(full_df[['CPI인플레이션_미국','CPI인플레이션_한국','CPI인플레이션_캐나다']], label = ['CPI인플레이션_미국','CPI인플레이션_한국','CPI인플레이션_캐나다'])
plt.legend()
plt.show()

plt.plot(full_df[['생산자물가지수_한국','생산자물가지수_미국','생산자물가지수_캐나다']].tail(12))
plt.show()

full_df[['생산자물가지수_한국','생산자물가지수_미국','생산자물가지수_캐나다']].tail(15)

# > 데이터가 관리되지 않는 듯하니.. 패스  

full_df['금리갭_캐나다한국'] = full_df['기준금리_캐나다'] - full_df['기준금리_한국']
full_df['금리갭_캐나다미국'] = full_df['기준금리_캐나다'] - full_df['기준금리_미국']
full_df['금리갭_미국한국'] = full_df['기준금리_미국'] - full_df['기준금리_한국']

# full_df['국채갭_캐나다한국'] = full_df['기준금리_캐나다'] - full_df['국채1년물_한국']
# full_df['국채갭_캐나다미국'] = full_df['기준금리_캐나다'] - full_df['국채1년물_미국']
# full_df['국채갭_미국한국'] = full_df['국채1년물_미국'] - full_df['국채1년물_한국']

full_df.columns

# ### 2-5) 외환보유금액  
# - 실질? 명목? 어떤것일지 몰라서 둘다 넣어두고 더 잘 설명되는 내용으로,  
# - 증가율도 , 월 단위 입수 데이터임   
# - 단위는 달러로 좀 맞추고 시작 

full_df['외환보유액_한국'] *= 1000000
full_df['외환보유액_캐나다'] *=1000000

# +
full_df['명목GDP대비외환보유액_한국'] = full_df['외환보유액_한국'] / full_df['명목GDP_한국']
full_df['명목GDP대비외환보유액_캐나다'] = full_df['외환보유액_캐나다'] / full_df['명목GDP_캐나다']

full_df['실질GDP대비외환보유액_한국'] = full_df['외환보유액_한국'] / full_df['실질GDP_한국']
full_df['실질GDP대비외환보유액_캐나다'] = full_df['외환보유액_캐나다'] / full_df['실질GDP_캐나다']

full_df['외환보유비중'] = full_df['외환보유액_한국']/full_df['외환보유액_캐나다']

full_df['외환보유증감_한국'] = full_df['외환보유액_한국'].pct_change(1)
full_df['외환보유증감_캐나다'] = full_df['외환보유액_캐나다'].pct_change(1)
# -

full_df.columns

# ### 2-6) 환율 정보 
# - ppp xr 조정 
# - 실질 환율 계산
# - 환율 모멘텀 추가?  

full_df['실질환율_KRWUSD'] = full_df['PPP_XR_KOR'] / full_df['환율_KRWUSD'] # indirect 니까 분자분모 이렇게 
full_df['실질환율_CADUSD'] = full_df['PPP_XR_CAN'] / full_df['환율_CADUSD'] # 1달러 = 1.29캐나다달러

full_df['실질환율_KRWCAD'] = full_df['실질환율_KRWUSD'] / full_df['실질환율_CADUSD']
full_df['실질환율_KRWCAD'].tail(3)

full_df[['실질환율_CADUSD','실질환율_KRWUSD','실질환율_KRWCAD']].tail()

full_df['환율_CADKRW_RET'] = full_df[['환율_CADKRW']].pct_change(1)
full_df['환율_CADUSD_RET'] = full_df[['환율_CADUSD']].pct_change(1)
full_df['환율_KRWUSD_RET'] = full_df[['환율_KRWUSD']].pct_change(1)

full_df.columns

# ### 2-7) 실업률 
# - 실업률이 왜 관계있는지?  

# ### 2-8) 조정완료된 항목 삭제  

remove_lst = ['GDP_미국','GDP_캐나다','GDP_한국',     
              'GDP_CURRENT_CAN','GDP_CURRENT_USA','GDP_CURRENT_KOR',                
              '명목GDP_미국','명목GDP_캐나다','명목GDP_한국',                
              '실질GDP_미국','실질GDP_캐나다','실질GDP_한국',
              '경상수지_미국','경상수지_캐나다','경상수지_한국',    
              '조정경상수지_미국','조정경상수지_캐나다','조정경상수지_한국',
              '소비자물가지수_미국','소비자물가지수_캐나다','소비자물가지수_한국',
              '생산자물가지수_미국','생산자물가지수_캐나다','생산자물가지수_한국',
              '외환보유액_캐나다','외환보유액_한국','환율_CADUSD','환율_KRWUSD','PPP_XR_KOR','PPP_XR_CAN','PPP_XR_USA',
              'GDP_CONSTANT_KOR', 'GDP_CONSTANT_CAN', 'GDP_CONSTANT_USA',
              'GDP_GROWTH_KOR', 'GDP_GROWTH_CAN', 'GDP_GROWTH_USA', ]
for c in remove_lst:
    try:
        full_df.drop(c, axis = 1, inplace = True)
    except:
        continue

full_df.columns

# ## 3. 목표변수 설정 
# - 12개월 수익률 대신 1 / 3 / 6 / 12 M 을 각자 진행

df = full_df.copy()

periods = 3
df['환율_RET'] = df['환율_CADKRW'].pct_change(periods = periods)
df[f'환율_RET{periods}'] = df['환율_RET'].shift(-periods)

plt.plot(df[f'환율_RET'])
plt.show()

df.drop(['환율_CADKRW','환율_RET'], axis = 1, inplace = True)

full_df.to_excel('full_df.xlsx')

# ## 4. EDA
# - 최소한의 eda만..

df.describe().T

# ### 4-1) ratio 지표는  %단위로

df.columns

fmt_conv_features = ['환율_RET3','환율_KRWUSD_RET','환율_CADUSD_RET','환율_CADKRW_RET',
                             'CPI인플레이션_한국','CPI인플레이션_미국','CPI인플레이션_캐나다']
for c in fmt_conv_features:
    df[c] = df[c]*100

# ### 4-2) 항목별 3개월 수익율과 관계 확인 

y = df['환율_RET3']
y_name = y.name
for c in df.columns:
    x = df[c]

    x_name = x.name
    sns.jointplot(x=x_name, y=y_name, data=df,
                      kind="reg", truncate=False,
                      color="m", height=5)
plt.show()

importances = df.drop('환율_RET3',1).apply(lambda x: x.corr(y)) #변수별 corr 계산하고 
indices = np.argsort(importances) # 정리해서 
corr_df = importances[indices] #결과에 넣고 
plt.figure(figsize=(10,10))
plt.title(f'{periods}개월 수익율과  상관계수')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), corr_df.index)
plt.xlabel('Relative Importance')
plt.show()

# > 경상수지 / 외환보유비중 / 기준금리 등이 영향도가 큰 것으로 보임  

# ### 4-3. Feature 간 관계

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin = -1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# +
i = 4
x_name = df.columns[i]
y = df['환율_RET3']
plt.figure(figsize = (12,2))
plt.plot(y, label = '수익률')
plt.plot(df[x_name], label = x_name)
plt.legend()
plt.show()

plt.figure(figsize = (12,2))
plt.plot((y-np.min(y))/(np.max(y) - np.min(y)), label = '수익률')
plt.plot((df[x_name]-np.min(df[x_name]))/(np.max(df[x_name]) - np.min(df[x_name])), label = x_name)
plt.legend()
plt.show()
# -

# ### 4-4. 변수별 p value

import statsmodels.api as sm

sample_df = df.copy()
sample_df.dropna(inplace = True)

X = sample_df.iloc[:,:-1]

y = sample_df.iloc[:,-1]

coef_df = {'Feature':[],'Coef':[],'R^2':[],'p-values':[]}

coef_df['Feature']

# 단일변수 회귀계수는?
coef_dict = {'Feature':[],'Coef':[],'R^2':[],'p-values':[]}
for c in X.columns:
    x = X[c]
    x = sm.add_constant(x)
    results = sm.OLS(y, x).fit()

    coef_dict['Feature'].append(c)
    coef_dict['Coef'].append(results.params[1])
    coef_dict['R^2'].append(results.rsquared  )
    coef_dict['p-values'].append(results.pvalues[1])
coef_df = pd.DataFrame(coef_dict)    

pd.options.display.float_format = '{:.4f}'.format
coef_df.sort_values(by = 'p-values')

# > 이중에서 일단 전혀 의미없어 보이는 것들은 좀 제외하고..  
# 변수 선정을 한 후에 모형으로 넘어가겠음  

# ##
