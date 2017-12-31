import os
import pandas as pd
import talib as ta
import numpy as np
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt


def Pricedata_load(min):

    start = time.time()

    os.chdir('C:/Users/tommy/Desktop/価格データ')
    pricedata = pd.read_csv('USDJPY'+str(min)+'.csv',encoding='shift-jis')
    colums = ['date','time','open','hi','low','close','volum']
    pricedata.columns = colums
    pricedata['datetime'] = pricedata.date + "  " + pricedata.time
    pricedata.index = pd.DatetimeIndex(pricedata.datetime)
    pricedata['RSI'] = ta.RSI(np.array(pricedata.close), timeperiod=14)
    pickle.dump(pricedata, open('USDJPY'+str(min)+'.dump', 'wb'))

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    return pricedata

def dow():
    os.chdir('C:/Users/tommy/Desktop/価格データ')
    dowdata = pd.read_csv('dow.csv')
    return dowdata

USDJPY15 = Pricedata_load(15)
USDJPY15 = pickle.load(open('USDJPY15'+'.dump','rb'))
USDJPY15['SMA_21'] = ta.SMA(np.array(USDJPY15.close),timeperiod=21)
USDJPY15.index +=  pd.offsets.Hour(7)
price = USDJPY15[USDJPY15.date>='2010']


# Dow = dow()
# Dowdata = Dow.rename(columns={'日付け':'date','終値':'close','始値':'open','高値':'hi','安値':'low','出来高':'volum','前日比%':'diff'})
# Dowdata['Dateate'] = Dowdata.date.str.replace('年','/')
# Dowdata['Dateate'] = Dowdata.date.str.replace('月','/')
# Dowdata['Date'] = Dowdata.date.str.replace('日','/')


tky_delt = pd.timedelta_range(start='02:00:00',end='09:45:00',freq='15T')
ldn_delt = pd.timedelta_range(start='10:00:00',end='17:45:00',freq='15T')
nyk_delt0 = pd.timedelta_range(start='18:00:00',end='23:45:00',freq='15T')
nyk_delt1 = pd.timedelta_range(start='00:00:00',end='01:45:00',freq='15T')


tky_colums = tky_delt.astype(object)
ldn_colums = ldn_delt.astype(object)
nyk_ob = nyk_delt0.astype(object)
nyk_ob1 = nyk_delt1.astype(object)

Tcolums = ['02:00','02:15','02:30','02:45',
           '03:00','03:15','03:30','03:45',
           '04:00','04:15','04:30','04:45',
           '05:00','05:15','05:30','05:45',
           '06:00','06:15','06:30','06:45',
           '07:00','07:15','07:30','07:45',
           '08:00','08:15','08:30','08:45',
           '09:00','09:15','09:30','09:45']
Lcolums = ['10:00','10:15','10:30','10:45',
           '11:00','11:15','11:30','11:45',
           '12:00','12:15','12:30','12:45',
           '12:00','12:15','12:30','12:45',
           '13:00','13:15','13:30','13:45',
           '14:00','14:15','14:30','14:45',
           '15:00','15:15','15:30','15:45',
           '16:00','16:15','16:30','16:45']
Ncolums = ['18:00','18:15','18:30','18:45',
           '19:00','19:15','19:30','19:45',
           '20:00','20:15','20:30','20:45',
           '21:00','21:15','21:30','21:45',
           '22:00','22:15','22:30','22:45',
           '23:00','23:15','23:30','23:45',
           '00:00','00:15','00:30','00:45',
           '01:00','01:15','01:30','01:45']

Tcolumsdf = pd.DataFrame(Tcolums)
Lcolumsdf = pd.DataFrame(Ncolums)
Ncolumsdf = pd.DataFrame(Ncolums)

from datetime import time
tkyprice = price[time(8,0):time(15,45)]
ldnprice = price[time(16,0):time(23,45)]
nykprice = price[time(0,0):time(7,45)]

daylength = price.close.resample('B').mean()

from bokeh.io import output_notebook
from bokeh.io import show




"データ構築"

def nyktky():

    from datetime import time
    import pandas.tseries.offsets as offsets

    TKY_open = price[time(8, 0):time(8, 0)]

    tky_open_hi    = np.zeros((len(TKY_open.index))
    tky_open_low   = np.zeros((len(TKY_open.index))
    tky_open_close = np.zeros((len(TKY_open.index))
    tky_hi_close   = np.zeros((len(TKY_open.index))
    tky_close_low  = np.zeros((len(TKY_open.index))
    tky_hi_low     = np.zeros((len(TKY_open.index))
    tky_close_low  = np.zeros((len(TKY_open.index))



    nyk_B = np.zeros((len(TKY_open), len(tky_delt)))
    tky_RSI = np.zeros(len(TKY_open))
    nyk_open_close = np.zeros(len(TKY_open))

    for (TO, Tcount) in zip(TKY_open.index, range(0, len(TKY_open), 1)):

        tkydata = price[TO:TO + offsets.Hour(7) + offsets.Minute(45)]
        nykdata = price[TO - offsets.Hour(8):TO - offsets.Minute(15)]
        try:
            tky_RSI[Tcount] = tkydata.RSI[0]
            nyk_open_close[Tcount] = ((nykdata.open[0] - nykdata.close[-1]) / nykdata.open[0]) * 100
        except:
            tky_RSI[Tcount] = (0)
            nyk_open_close[Tcount] = (0)

        tkyopen = tkydata.open[0]
        nykclose = nykdata.close

        for (TC, TCount, RSI, nc) in zip(tkydata.close, range(0, len(tkydata), 1), tkydata.RSI, nykdata.close):
            try:
                tky_F[Tcount, TCount] = ((TC - tkyopen) / tkyopen) * 100
                nyk_B[Tcount, TCount] = ((tkyopen - nc) / tkyopen) * 100
            except:
                tky_F[Tcount, TCount] = (0)
                nyk_B[Tcount, TCount] = (0)

    df_tkyF = pd.DataFrame(tky_F, index=TKY_open.index, columns=Tcolumsdf[0])
    df_nykB = pd.DataFrame(nyk_B, index=TKY_open.index, columns=Ncolumsdf[0])
    df_tky_RSI = pd.DataFrame(tky_RSI, index=TKY_open.index, columns=['TKY_RSI'])
    df_tky_RSI1 = df_tky_RSI.fillna(0)
    df_nyk_open_close = pd.DataFrame(nyk_open_close, index=TKY_open.index, columns=['NYK_diff'])
    NY_TKY = pd.concat([df_nykB, df_tkyF, df_tky_RSI1, df_nyk_open_close], axis=1)

    return NY_TKY
nyktky_diff = nyktky()


"クラスタリング"
def clastering(clusters):
    from sklearn.cluster import KMeans
    from sklearn.model_selection import  train_test_split

    X= nyktky_diff.iloc[:,0:66].values
    # X= NY_TKY.iloc[:,np.r_[65,32:64]].values
    X_train = X

    import time
    # clusters  = 10
    start_watch = time.time()

    km = KMeans(n_clusters=clusters,
               init='k-means++',
               n_init=10,
               max_iter=300,
               tol=1e-04,
               random_state=1)

    y_km = km.fit_predict( X_train )

    elapsed_time = time.time() - start_watch
    print('処理時間：',elapsed_time,'秒')

    nyktky_diff['Label'] = y_km
    import dill
    dill.dump(km, open('km.cmp', 'wb'))

    return  nyktky_diff
nyktky_diff_onLabel = clastering(10)

"プロット関数"
def plotclastering(clusters):

    clu_valu = nyktky_diff_onLabel.Label.value_counts()
    df_clu_valu = pd.DataFrame(clu_valu)

    # plt.axhline(0,color='k',lw=2)
    # plt.axvline(color='red',lw=2)
    # plt.plot(NY_TKY.iloc[:,:64].T)
    nyktky_diff_onLabel.iloc[:, np.r_[0:63]].T.plot(figsize=(18, 8), legend=False, grid=True, title='ALLDATA')
    plt.axhline(0, color='k', lw=2)
    # plt.show()

    for c in range(0, clusters, 1):
        clu_count = df_clu_valu[df_clu_valu.index == c]
        # print(clu_count)
        # plt.title("AfterTrend"+str(clu_count))
        # plt.axhline(0,color='k',lw=2)
        nyktky_diff_onLabel[nyktky_diff_onLabel.Label == c].iloc[:, np.r_[0:63]].T.plot(figsize=(18, 8), legend=False, grid=True,title='DATA' + str(clu_count))
        plt.axhline(0, color='k', lw=2)
        # plt.xlabel('After_Minute')
        # plt.ylabel('Diff')
        plt.show()
# plotclastering(20)

"ランダムフォレスト"
def nyktky_forest(forestcount):


    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X= nyktky_diff_onLabel.iloc[:,:66].values
    # X= NY_TKY2.NYK_diff
    y= nyktky_diff_onLabel.Label.values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

    forest = RandomForestClassifier(criterion='entropy',
                                   n_estimators=forestcount,
                                   random_state=1,
                                    n_jobs=2)
    forest.fit(X_train,y_train)

    "予測データの作成"
    y_predict = forest.predict(X_test)
    "正解率"
    print("Accurancy onn training set_訓練セットの精度: {:.3f}".format(forest.score(X_train,y_train)))
    print("Accurancy onn training set_テストセットの精度: {:.3f}".format(forest.score(X_test,y_test)))
nyktky_forest(11)

"交差検証"
def crossveri():
    from distutils.version import LooseVersion as Version
    from sklearn import __version__ as sklearn_version
    if Version(sklearn_version) < '0.18':
        from sklearn.cross_validation import train_test_split
    else:
        from sklearn.model_selection import train_test_split


    if Version(sklearn_version) < '0.18':
        from sklearn.cross_validation import StratifiedKFold
    else:
        from sklearn.model_selection import StratifiedKFold

    "K分割交差検証でランダムフォレストモデルの汎化性能を評価する"
    if Version(sklearn_version)<'0.18':
        kfold = StratifiedKFold(y=y_train, n_folds=10,random_state=1)
    else:
        kfold = StratifiedKFold(n_splits=10,
                                random_state=1).split(X_train,y_train)

    scores = []
    for k,(train,test) in enumerate(kfold):
        forest.fit(X_train[train],y_train[train])
        score = forest.score(X_train[test],y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s,Acc: %.3f' % (k+1,np.bincount(y_train[train]),score))
    print('\nCV accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))
crossveri()

def forestimportant():
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    feat_labels = nyktky_diff_onLabel.columns[0:66]

    plt.figure(figsize=(18, 5))
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')
    plt.xticks(range(X_train.shape[1]),
               feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

forestimportant()



Dow.head()






