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

import FinanceDataReader as fdr
import pandas as pd

# #### 1) 명목환율 확보  

df_cadkrw = fdr.DataReader('CAD/KRW', '1995')
df_cadusd = fdr.DataReader('CAD/USD', '1995')
df_usdkrw = fdr.DataReader('USD/KRW', '1995')

df_cadkrw.to_excel('./data/fds/df_cadkrw.xlsx')
df_cadusd.to_excel('./data/fds/df_cadusd.xlsx')
df_usdkrw.to_excel('./data/fds/df_usdkrw.xlsx')

# #### 2) World bank Data

#
# https://data.worldbank.org/indicator/PA.NUS.PPPC.RF?locations=CA

# ppp 환율
df_ppp = pd.read_excel('./data/wb/PPP_XR.xls')
# 실질환율
df_realxr = pd.read_excel('./data/wb/REAL_XR.xls')
# gdp
df_gdp = pd.read_excel('./data/wb/GDP.xls')
# gdp 연증가율
df_gdp_g = pd.read_excel('./data/wb/GDP_growth.xls')
# 인구증가율대비gdp 증가율
df_gdp_pcg = pd.read_excel('./data/wb/GDP_per_capita_growth.xls')

# ㅇ 2019.2분기 기준 캐나다 경제는 서비스업 71% 및 제조업 29%로 구분되며, ①부동산 산업(12.7%) ②제조업 (10.3%) ③에너지 산업 (9.2%) ④건설업 (7.0%), 금융·보험 산업(6.6%) 등이 경제에서 차지하는 비중이 높음.

# https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_2KAA906_OECD

# #### 3) Deep Search Data

import os

os.listdir('./data/ds')

# > 확보 정보  
# 환율(한국/미국/캐나다 3cross)
# 경상수지(한국/미국/캐나다) => 수입/수출 분리도 가능할듯 ?  
# 생산자물가지수(한국/미국/캐나다)  => 총지수가 해당 내용인지 확인 필요  
# 소비자물가지수(한국/미국/캐나다)  
# 실업율(한국/미국/캐나다)  
# 기준금리(한국/미국/캐나다)  
# 국고채 1년물(한국/미국/-) => 뭐라 찾을지 모르겠음  
# 금 현물.. 이건 나라별로 안되나?  
# 서비스수지(-/-/캐나다) => 대부분 주요 gdp비중이 서비스업이라.. 찾아볼까 했는데 의미 없을듯  
# 외환보유액(한국/캐나다/-/중국)  => 그냥 중국 넣어봄  
# 주요무역국의 거시지표를 포함시키면?(EU/중국)  
#
#
