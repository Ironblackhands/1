import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas.tseries.offsets as offsets
from datetime import time
import talib as ta
import pickle
import seaborn as sns
from tqdm import tqdm

pricefold = 'C:/Users/tommy/Desktop/価格データ'

dumpfold = 'C:/Users/tommy/Desktop/価格データ/dump'

st = '2010.01.01'
ed = '2017.12.22'


def dataimport(path, st, ed):
    """

    :param path:
    :param st:開始日
    :param end:終了日
    :return:
    """
    "データ読み込みフォルダ指定"
    os.chdir(path)
    pricedata = pd.read_csv('USDJPY15.csv')
    colums = ['date', 'time', 'open', 'hi', 'low', 'close', 'volume']
    pricedata.columns = colums
    pricedata['datetime'] = pricedata.date + "  " + pricedata.time
    pricedata1 = pricedata[(pricedata.date >= st) & (pricedata.date <= ed)]
    pricedata1.index = pd.DatetimeIndex(pricedata1.datetime)
    pricedata1['RSI'] = ta.RSI(np.array(pricedata1.close), timeperiod=14)
    pricedata1['EMA'] = ta.EMA(np.array(pricedata1.close), timeperiod=64)
    pricedata1['EMA_devi'] = ((pricedata1.close - pricedata1.EMA) / pricedata1.EMA) * 100

    os.chdir(dumpfold)
    pickle.dump(pricedata1, open('USDJPY' + '.dump', 'wb'))

    return pricedata1


#
pricedata = pd.DataFrame()
pricedata = dataimport(pricefold, st, ed)

"dumpファイル読み込み"
os.chdir(dumpfold)
pricedata = pickle.load(open('USDJPY' + '.dump', 'rb'))

price = pricedata['2010-01-01':'2017-12-22']

"東京市場の高値安値時間の調査"
tkyprice = price[time(2, 0):time(9, 45)]
datarange = pd.date_range(start=tkyprice.index[0], end=tkyprice.index[-1], freq='B')

tky_delt = pd.timedelta_range(start='02:00:00', end='09:45:00', freq='15T')
nyk_delt0 = pd.timedelta_range(start='18:00:00', end='23:45:00', freq='15T')
nyk_delt1 = pd.timedelta_range(start='00:00:00', end='01:45:00', freq='15T')
tky_colums = tky_delt.astype(object)
nyk_ob = nyk_delt0.astype(object)
nyk_ob1 = nyk_delt1.astype(object)

Tcolums = ['02:00', '02:15', '02:30', '02:45',
           '03:00', '03:15', '03:30', '03:45',
           '04:00', '04:15', '04:30', '04:45',
           '05:00', '05:15', '05:30', '05:45',
           '06:00', '06:15', '06:30', '06:45',
           '07:00', '07:15', '07:30', '07:45',
           '08:00', '08:15', '08:30', '08:45',
           '09:00', '09:15', '09:30', '09:45']

Ncolums = ['18:00', '18:15', '18:30', '18:45',
           '19:00', '19:15', '19:30', '19:45',
           '20:00', '20:15', '20:30', '20:45',
           '21:00', '21:15', '21:30', '21:45',
           '22:00', '22:15', '22:30', '22:45',
           '23:00', '23:15', '23:30', '23:45',
           '00:00', '00:15', '00:30', '00:45',
           '01:00', '01:15', '01:30', '01:45']

Tcolumsdf = pd.DataFrame(Tcolums)
Ncolumsdf = pd.DataFrame(Ncolums)

np_days = np.zeros(len(datarange))


def makedata():
    from datetime import time
    TKY_open = price[time(2, 0):time(2, 0)]

    tky_F = np.zeros((len(TKY_open.index), len(tky_delt)))
    np_tky_diff = np.zeros(len(TKY_open))
    np_nyk_diff = np.zeros(len(TKY_open))
    np_tky_hi_time = np.zeros(len(TKY_open))
    np_tky_low_time = np.zeros(len(TKY_open))
    np_nyk_hi_time = np.zeros(len(TKY_open))
    np_low_time = np.zeros(len(TKY_open))

    np_nhi_low = np.zeros(len(TKY_open))
    np_nhi_close = np.zeros(len(TKY_open))
    np_nclose_low = np.zeros(len(TKY_open))
    np_nhi_open = np.zeros(len(TKY_open))
    np_nopen_low = np.zeros(len(TKY_open))

    np_tky_open_RSI = np.zeros(len(TKY_open))
    np_tky_EMA_devi = np.zeros(len(TKY_open))

    for (TO, Tcount) in tqdm(zip(TKY_open.index, range(0, len(TKY_open), 1))):
        tkydata = price[TO:TO + offsets.Hour(7) + offsets.Minute(45)]

        if tkydata.index[0].weekday() == 0:
            nykdata = price[TO - offsets.Day(2) - offsets.Hour(8):TO - offsets.Minute(15)]

            np_nyk_hi_time[Tcount] = (
                    nykdata[nykdata.hi == nykdata.iloc[:20].hi.max()].index[0] - nykdata.open.index[0]).total_seconds()
            np_low_time[Tcount] = (
                    nykdata[nykdata.low == nykdata.iloc[:20].low.min()].index[0] - nykdata.open.index[
                0]).total_seconds()

            np_nhi_low[Tcount] = nykdata.iloc[:20].hi.max() - nykdata.iloc[:20].low.min()
            np_nhi_close[Tcount] = nykdata.iloc[:20].hi.max() - nykdata.iloc[:20].close[-1]
            np_nclose_low[Tcount] = nykdata.iloc[:20].close[-1] - nykdata.iloc[:20].low.min()
            np_nhi_open[Tcount] = nykdata.iloc[:20].hi.max() - nykdata.iloc[:20].open[0]
            np_nopen_low[Tcount] = nykdata.iloc[:20].open[0] - nykdata.iloc[:20].low.min()

        else:
            nykdata = price[TO - offsets.Hour(8):TO - offsets.Minute(15)]

            np_nyk_hi_time[Tcount] = (
                    nykdata[nykdata.hi == nykdata.hi.max()].index[0] - nykdata.open.index[0]).total_seconds()
            np_low_time[Tcount] = (
                    nykdata[nykdata.hi == nykdata.hi.min()].index[0] - nykdata.open.index[0]).total_seconds()

            np_nhi_low[Tcount] = nykdata.hi.max() - nykdata.low.min()
            np_nhi_close[Tcount] = nykdata.hi.max() - nykdata.close[-1]
            np_nclose_low[Tcount] = nykdata.close[-1] - nykdata.low.min()
            np_nhi_open[Tcount] = nykdata.hi.max() - nykdata.open[0]
            np_nopen_low[Tcount] = nykdata.open[0] - nykdata.low.min()

        tkyopen = tkydata.open[0]
        tkyclose = tkydata.close[-1]

        nykopen = nykdata.open[0]
        nykclose = nykdata.close[-1]

        np_tky_diff[[Tcount]] = np.log(tkyclose) - np.log(tkyopen)
        np_nyk_diff[[Tcount]] = np.log(nykclose) - np.log(nykopen)
        np_tky_open_RSI[[Tcount]] = tkydata.RSI[0]
        np_tky_EMA_devi[[Tcount]] = tkydata.EMA_devi[0]

        np_tky_hi_time[Tcount] = (
                tkydata[tkydata.hi == tkydata.hi.max()].index[0] - tkydata.open.index[0]).total_seconds()
        np_tky_low_time[Tcount] = (
                tkydata[tkydata.hi == tkydata.hi.min()].index[0] - tkydata.open.index[0]).total_seconds()

    df_tky_diff = pd.DataFrame(np_tky_diff, index=TKY_open.index, columns=['tkydiff'])
    df_tky_hi_time = pd.DataFrame(np_tky_hi_time, index=TKY_open.index, columns=['hi_time'])
    df_tky_low_time = pd.DataFrame(np_tky_low_time, index=TKY_open.index, columns=['low_time'])
    df_tky_open_RSI = pd.DataFrame(np_tky_open_RSI, index=TKY_open.index, columns=['TRSI'])
    df_tky_EMA_devi = pd.DataFrame(np_tky_EMA_devi,index=TKY_open.index,columns=['EMA_devi'])

    df_nyk_diff = pd.DataFrame(np_nyk_diff, index=TKY_open.index, columns=['nykdiff'])
    df_nyk_hitime = pd.DataFrame(np_nyk_hi_time, index=TKY_open.index, columns=['nhi_time'])
    df_nyk_lowtime = pd.DataFrame(np_low_time, index=TKY_open.index, columns=['nlow_time'])
    df_nhi_low = pd.DataFrame(np_nhi_low, index=TKY_open.index, columns=['nH_L'])
    df_nhi_close = pd.DataFrame(np_nhi_close, index=TKY_open.index, columns=['nH_C'])
    df_nclose_low = pd.DataFrame(np_nclose_low, index=TKY_open.index, columns=['nC_L'])
    df_nhi_open = pd.DataFrame(np_nhi_open, index=TKY_open.index, columns=['nH_O'])
    df_nopen_low = pd.DataFrame(np_nopen_low, index=TKY_open.index, columns=['nO_L'])

    tky_features = pd.concat(
        [df_nhi_low, df_nhi_close, df_nclose_low, df_nhi_open, df_nopen_low, df_nyk_diff, df_nyk_hitime, df_nyk_lowtime,
         df_tky_hi_time, df_tky_low_time, df_tky_open_RSI,df_tky_EMA_devi, df_tky_diff], axis=1)

    return tky_features


TKY_open = price[time(2, 0):time(2, 0)]
featuresdata = pd.DataFrame()
featuresdata = makedata()
featuresdata.index = featuresdata.index.round('D')


pricedata.head()
