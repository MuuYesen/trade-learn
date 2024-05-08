# 以下所有函数如无特别说明，输入参数S均为numpy序列或者列表list，N为整型int
# 应用层1级函数完美兼容通达信或同花顺，具体使用方法请参考通达信

import math
import numpy as np
import pandas as pd


# ------------------ 0级：核心工具函数 --------------------------------------------
def RD(N, D=3):   return np.round(N, D)  # 四舍五入取3位小数


def RET(S, N=1):  return np.array(S)[-N]  # 返回序列倒数第N个值,默认返回最后一个


def ABS(S):      return np.abs(S)  # 返回N的绝对值


def LN(S):       return np.log(S)  # 求底是e的自然对数,


def POW(S, N):    return np.power(S, N)  # 求S的N次方


def SQRT(S):     return np.sqrt(S)  # 求S的平方根


def SIN(S):      return np.sin(S)  # 求S的正弦值（弧度)


def COS(S):      return np.cos(S)  # 求S的余弦值（弧度)


def TAN(S):      return np.tan(S)  # 求S的正切值（弧度)


def MAX(S1, S2):  return np.maximum(S1, S2)  # 序列max


def MIN(S1, S2):  return np.minimum(S1, S2)  # 序列min


def IF(S, A, B):   return np.where(S, A, B)  # 序列布尔判断 return=A  if S==True  else  B


def REF(S, N=1):  # 对序列整体下移动N,返回序列(shift后会产生NAN)
    return pd.Series(S).shift(N).values


def DIFF(S, N=1):  # 前一个值减后一个值,前面会产生nan
    return pd.Series(S).diff(N).values  # np.diff(S)直接删除nan，会少一行


def STD(S, N):  # 求序列的N日标准差，返回序列
    return pd.Series(S).rolling(N).std(ddof=0).values


def SUM(S, N):  # 对序列求N天累计和，返回序列    N=0对序列所有依次求和
    return pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values


def CONST(S):  # 返回序列S最后的值组成常量序列
    return np.full(len(S), S[-1])


def HHV(S, N):  # HHV(C, 5) 最近5天收盘最高价
    return pd.Series(S).rolling(N).max().values


def LLV(S, N):  # LLV(C, 5) 最近5天收盘最低价
    return pd.Series(S).rolling(N).min().values


def HHVBARS(S, N):  # 求N周期内S最高值到当前周期数, 返回序列
    return pd.Series(S).rolling(N).apply(lambda x: np.argmax(x[::-1]), raw=True).values


def LLVBARS(S, N):  # 求N周期内S最低值到当前周期数, 返回序列
    return pd.Series(S).rolling(N).apply(lambda x: np.argmin(x[::-1]), raw=True).values


def MA(S, N):  # 求序列的N日简单移动平均值，返回序列
    return pd.Series(S).rolling(N).mean().values


def EMA(S, N):  # 指数移动平均,为了精度 S>4*N  EMA至少需要120周期     alpha=2/(span+1)
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


def SMA(S, N, M=1):  # 中国式的SMA,至少需要120周期才精确 (雪球180周期)    alpha=1/(1+com)
    return pd.Series(S).ewm(alpha=M / N, adjust=False).mean().values  # com=N-M/M


def WMA(S, N):  # 通达信S序列的N日加权移动平均 Yn = (1*X1+2*X2+3*X3+...+n*Xn)/(1+2+3+...+Xn)
    return pd.Series(S).rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1), raw=True).values


def DMA(S, A):  # 求S的动态移动平均，A作平滑因子,必须 0<A<1  (此为核心函数，非指标）
    if isinstance(A, (int, float)):  return pd.Series(S).ewm(alpha=A, adjust=False).mean().values
    A = np.array(A);
    A[np.isnan(A)] = 1.0;
    Y = np.zeros(len(S));
    Y[0] = S[0]
    for i in range(1, len(S)): Y[i] = A[i] * S[i] + (1 - A[i]) * Y[i - 1]  # A支持序列 by jqz1226
    return Y


def AVEDEV(S, N):  # 平均绝对偏差  (序列与其平均值的绝对差的平均值)
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values


def SLOPE(S, N):  # 返S序列N周期回线性回归斜率
    return pd.Series(S).rolling(N).apply(lambda x: np.polyfit(range(N), x, deg=1)[0], raw=True).values


def FORCAST(S, N):  # 返回S序列N周期回线性回归后的预测值， jqz1226改进成序列出
    return pd.Series(S).rolling(N).apply(lambda x: np.polyval(np.polyfit(range(N), x, deg=1), N - 1), raw=True).values


def LAST(S, A, B):  # 从前A日到前B日一直满足S_BOOL条件, 要求A>B & A>0 & B>=0
    return np.array(pd.Series(S).rolling(A + 1).apply(lambda x: np.all(x[::-1][B:]), raw=True), dtype=bool)


# ------------------   1级：应用层函数(通过0级核心函数实现）使用方法请参考通达信--------------------------------
def COUNT(S, N):  # COUNT(CLOSE>O, N):  最近N天满足S_BOO的天数  True的天数
    return SUM(S, N)


def EVERY(S, N):  # EVERY(CLOSE>O, 5)   最近N天是否都是True
    return IF(SUM(S, N) == N, True, False)


def EXIST(S, N):  # EXIST(CLOSE>3010, N=5)  n日内是否存在一天大于3000点
    return IF(SUM(S, N) > 0, True, False)


def FILTER(S, N):  # FILTER函数，S满足条件后，将其后N周期内的数据置为0, FILTER(C==H,5)
    for i in range(len(S)): S[i + 1:i + 1 + N] = 0 if S[i] else S[i + 1:i + 1 + N]
    return S  # 例：FILTER(C==H,5) 涨停后，后5天不再发出信号


def BARSLAST(S):  # 上一次条件成立到当前的周期, BARSLAST(C/REF(C,1)>=1.1) 上一次涨停到今天的天数
    M = np.concatenate(([0], np.where(S, 1, 0)))
    for i in range(1, len(M)):  M[i] = 0 if M[i] else M[i - 1] + 1
    return M[1:]


def BARSLASTCOUNT(S):  # 统计连续满足S条件的周期数        by jqz1226
    rt = np.zeros(len(S) + 1)  # BARSLASTCOUNT(CLOSE>OPEN)表示统计连续收阳的周期数
    for i in range(len(S)): rt[i + 1] = rt[i] + 1 if S[i] else rt[i + 1]
    return rt[1:]


def BARSSINCEN(S, N):  # N周期内第一次S条件成立到现在的周期数,N为常量  by jqz1226
    return pd.Series(S).rolling(N).apply(lambda x: N - 1 - np.argmax(x) if np.argmax(x) or x[0] else 0,
                                         raw=True).fillna(0).values.astype(int)


def CROSS(S1, S2):  # 判断向上金叉穿越 CROSS(MA(C,5),MA(C,10))  判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # 不使用0级函数,移植方便  by jqz1226


def LONGCROSS(S1, S2, N):  # 两条线维持一定周期后交叉,S1在N周期内都小于S2,本周期从S1下方向上穿过S2时返回1,否则返回0
    return np.array(np.logical_and(LAST(S1 < S2, N, 1), (S1 > S2)), dtype=bool)  # N=1时等同于CROSS(S1, S2)


def VALUEWHEN(S, X):  # 当S条件成立时,取X的当前值,否则取VALUEWHEN的上个成立时的X值   by jqz1226
    return pd.Series(np.where(S, X, np.nan)).ffill().values


def BETWEEN(S, A, B):  # S处于A和B之间时为真。 包括 A<S<B 或 A>S>B
    return ((A < S) & (S < B)) | ((A > S) & (S > B))


def TOPRANGE(S):  # TOPRANGE(HIGH)表示当前最高价是近多少周期内最高价的最大值 by jqz1226
    rt = np.zeros(len(S))
    for i in range(1, len(S)):  rt[i] = np.argmin(np.flipud(S[:i] < S[i]))
    return rt.astype('int')


def LOWRANGE(S):  # LOWRANGE(LOW)表示当前最低价是近多少周期内最低价的最小值 by jqz1226
    rt = np.zeros(len(S))
    for i in range(1, len(S)):  rt[i] = np.argmin(np.flipud(S[:i] > S[i]))
    return rt.astype('int')

# ------------------------工具函数---------------------------------------------

def HHV(S, N):  # HHV,支持N为序列版本
    # type: (np.ndarray, Optional[int,float, np.ndarray]) -> np.ndarray
    """
    HHV(C, 5)  # 最近5天收盘最高价
    """
    if isinstance(N, (int, float)):
        return pd.Series(S).rolling(N).max().values
    else:
        res = np.repeat(np.nan, len(S))
        for i in range(len(S)):
            if (not np.isnan(N[i])) and N[i] <= i + 1:
                res[i] = S[i + 1 - N[i]:i + 1].max()
        return res


def LLV(S, N):  # LLV,支持N为序列版本
    # type: (np.ndarray, Optional[int,float, np.ndarray]) -> np.ndarray
    """
    LLV(C, 5)  # 最近5天收盘最低价
    """
    if isinstance(N, (int, float)):
        return pd.Series(S).rolling(N).min().values
    else:
        res = np.repeat(np.nan, len(S))
        for i in range(len(S)):
            if (not np.isnan(N[i])) and N[i] <= i + 1:
                res[i] = S[i + 1 - N[i]:i + 1].min()
        return res


def DSMA(X, N):  # 偏差自适应移动平均线   type: (np.ndarray, int) -> np.ndarray
    """
    Deviation Scaled Moving Average (DSMA)    Python by: jqz1226, 2021-12-27
    Referred function from myTT: SUM, DMA
    """
    a1 = math.exp(- 1.414 * math.pi * 2 / N)
    b1 = 2 * a1 * math.cos(1.414 * math.pi * 2 / N)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    Zeros = np.pad(X[2:] - X[:-2], (2, 0), 'constant')
    Filt = np.zeros(len(X))
    for i in range(len(X)):
        Filt[i] = c1 * (Zeros[i] + Zeros[i - 1]) / 2 + c2 * Filt[i - 1] + c3 * Filt[i - 2]

    RMS = np.sqrt(SUM(np.square(Filt), N) / N)
    ScaledFilt = Filt / RMS
    alpha1 = np.abs(ScaledFilt) * 5 / N
    return DMA(X, alpha1)


def SUMBARSFAST(X, A):
    # type: (np.ndarray, Optional[np.ndarray, float, int]) -> np.ndarray
    """
    通达信SumBars函数的Python实现  by jqz1226
    SumBars函数将X向前累加，直到大于等于A, 返回这个区间的周期数。例如SUMBARS(VOL, CAPITAL),求完全换手的周期数。
    :param X: 数组。被累计的源数据。 源数组中不能有小于0的元素。
    :param A: 数组（一组）或者浮点数（一个）或者整数（一个），累加截止的界限数
    :return:  数组。各K线分别对应的周期数
    """
    if any(X <= 0):   raise ValueError('数组X的每个元素都必须大于0！')

    X = np.flipud(X)  # 倒转
    length = len(X)

    if isinstance(A * 1.0, float):  A = np.repeat(A, length)  # 是单值则转化为数组
    A = np.flipud(A)  # 倒转
    sumbars = np.zeros(length)  # 初始化sumbars为0
    Sigma = np.insert(np.cumsum(X), 0, 0.0)  # 在累加值前面插入一个0.0（元素变多1个，便于引用）

    for i in range(length):
        k = np.searchsorted(Sigma[i + 1:], A[i] + Sigma[i])
        if k < length - i:  # 找到
            sumbars[length - i - 1] = k + 1
    return sumbars.astype(int)


class Tdx30:
    def __init__(self, df_data):
        self.open = df_data['open'] # 开盘价
        self.high = df_data['high'] # 最高价
        self.low = df_data['low'] # 最低价
        self.close = df_data['close'] # 收盘价
        self.volumeume = df_data['volume'] # 成交量

    def MACD(self, SHORT=12, LONG=26, M=9):  # EMA的关系，S取120日，和雪球小数点2位相同
        DIF = EMA(self.close, SHORT) - EMA(self.close, LONG);
        DEA = EMA(DIF, M);
        MACD = (DIF - DEA) * 2
        return RD(DIF), RD(DEA), RD(MACD)

    def KDJ(self, N=9, M1=3, M2=3):  # KDJ指标
        RSV = (self.close - LLV(self.low, N)) / (HHV(self.high, N) - LLV(self.low, N)) * 100
        K = EMA(RSV, (M1 * 2 - 1));
        D = EMA(K, (M2 * 2 - 1));
        J = K * 3 - D * 2
        return K, D, J

    def RSI(self, N=24):  # RSI指标,和通达信小数点2位相同
        DIF = self.close - REF(self.close, 1)
        return RD(SMA(MAX(DIF, 0), N) / SMA(ABS(DIF), N) * 100)

    def WR(self, N=10, N1=6):  # W&R 威廉指标
        WR = (HHV(self.high, N) - self.close) / (HHV(self.high, N) - LLV(self.low, N)) * 100
        WR1 = (HHV(self.high, N1) - self.close) / (HHV(self.high, N1) - LLV(self.low, N1)) * 100
        return RD(WR), RD(WR1)

    def BIAS(self, L1=6, L2=12, L3=24):  # BIAS乖离率
        BIAS1 = (self.close - MA(self.close, L1)) / MA(self.close, L1) * 100
        BIAS2 = (self.close - MA(self.close, L2)) / MA(self.close, L2) * 100
        BIAS3 = (self.close - MA(self.close, L3)) / MA(self.close, L3) * 100
        return RD(BIAS1), RD(BIAS2), RD(BIAS3)

    def BOLL(self, N=20, P=2):  # BOLL指标，布林带
        MID = MA(self.close, N);
        UPPER = MID + STD(self.close, N) * P
        self.lowER = MID - STD(self.close, N) * P
        return RD(UPPER), RD(MID), RD(self.lowER)

    def PSY(self, N=12, M=6):
        PSY = COUNT(self.close > REF(self.close, 1), N) / N * 100
        PSYMA = MA(PSY, M)
        return RD(PSY), RD(PSYMA)

    def CCI(self, N=14):
        TP = (self.high + self.low + self.close) / 3
        return (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))

    def ATR(self, N=20):  # 真实波动N日平均值
        TR = MAX(MAX((self.high - self.low), ABS(REF(self.close, 1) - self.high)), ABS(REF(self.close, 1) - self.low))
        return MA(TR, N)

    def BBI(self, M1=3, M2=6, M3=12, M4=20):  # BBI多空指标
        return (MA(self.close, M1) + MA(self.close, M2) + MA(self.close, M3) + MA(self.close, M4)) / 4

    def DMI(self, M1=14, M2=6):  # 动向指标：结果和同花顺，通达信完全一致
        TR = SUM(MAX(MAX(self.high - self.low, ABS(self.high - REF(self.close, 1))), ABS(self.low - REF(self.close, 1))), M1)
        HD = self.high - REF(self.high, 1);
        LD = REF(self.low, 1) - self.low
        DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
        DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
        PDI = DMP * 100 / TR;
        MDI = DMM * 100 / TR
        ADX = MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
        ADXR = (ADX + REF(ADX, M2)) / 2
        return PDI, MDI, ADX, ADXR

    def TAQ(self, N):  # 唐安奇通道(海龟)交易指标，大道至简，能穿越牛熊
        UP = HHV(self.high, N);
        DOWN = LLV(self.low, N);
        MID = (UP + DOWN) / 2
        return UP, MID, DOWN

    def KTN(self, N=20, M=10):  # 肯特纳交易通道, N选20日，ATR选10日
        MID = EMA((self.high + self.low + self.close) / 3, N)
        ATRN = self.ATR(self.close, self.high, self.low, M)
        UPPER = MID + 2 * ATRN;
        self.lowER = MID - 2 * ATRN
        return UPPER, MID, self.lowER

    def TRIX(self, M1=12, M2=20):  # 三重指数平滑平均线
        TR = EMA(EMA(EMA(self.close, M1), M1), M1)
        TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
        TRMA = MA(TRIX, M2)
        return TRIX, TRMA

    def VR(self, M1=26):  # VR容量比率
        LC = REF(self.close, 1)
        return SUM(IF(self.close > LC, self.volume, 0), M1) / SUM(IF(self.close <= LC, self.volume, 0), M1) * 100

    def CR(self, N=20):  # CR价格动量指标
        MID = REF(self.high + self.low + self.close, 1) / 3;
        return SUM(MAX(0, self.high - MID), N) / SUM(MAX(0, MID - self.low), N) * 100

    def EMV(self, N=14, M=9):  # 简易波动指标
        self.volumeUME = MA(self.volume, N) / self.volume;
        MID = 100 * (self.high + self.low - REF(self.high + self.low, 1)) / (self.high + self.low)
        EMV = MA(MID * self.volumeUME * (self.high - self.low) / MA(self.high - self.low, N), N);
        MAEMV = MA(EMV, M)
        return EMV, MAEMV

    def DPO(self, M1=20, M2=10, M3=6):  # 区间震荡线
        DPO = self.close - REF(MA(self.close, M1), M2);
        MADPO = MA(DPO, M3)
        return DPO, MADPO

    def BRAR(self, M1=26):  # BRAR-ARBR 情绪指标
        AR = SUM(self.high - self.open, M1) / SUM(self.open - self.low, M1) * 100
        BR = SUM(MAX(0, self.high - REF(self.close, 1)), M1) / SUM(MAX(0, REF(self.close, 1) - self.low), M1) * 100
        return AR, BR

    def DFMA(self, N1=10, N2=50, M=10):  # 平行线差指标
        DIF = MA(self.close, N1) - MA(self.close, N2);
        DIFMA = MA(DIF, M)  # 通达信指标叫DMA 同花顺叫新DMA
        return DIF, DIFMA

    def MTM(self, N=12, M=6):  # 动量指标
        MTM = self.close - REF(self.close, N);
        MTMMA = MA(MTM, M)
        return MTM, MTMMA

    def MASS(self, N1=9, N2=25, M=6):  # 梅斯线
        MASS = SUM(MA(self.high - self.low, N1) / MA(MA(self.high - self.low, N1), N1), N2)
        MA_MASS = MA(MASS, M)
        return MASS, MA_MASS

    def ROC(self, N=12, M=6):  # 变动率指标
        ROC = 100 * (self.close - REF(self.close, N)) / REF(self.close, N);
        MAROC = MA(ROC, M)
        return ROC, MAROC

    def EXPMA(self, N1=12, N2=50):  # EMA指数平均数指标
        return EMA(self.close, N1), EMA(self.close, N2);

    def OBV(self):  # 能量潮指标
        return SUM(IF(self.close > REF(self.close, 1), self.volume, IF(self.close < REF(self.close, 1), -self.volume, 0)), 0) / 10000

    def MFI(self, N=14):  # MFI指标是成交量的RSI指标
        TYP = (self.high + self.low + self.close) / 3
        V1 = SUM(IF(TYP > REF(TYP, 1), TYP * self.volume, 0), N) / SUM(IF(TYP < REF(TYP, 1), TYP * self.volume, 0), N)
        return 100 - (100 / (1 + V1))

    def ASI(self, M1=26, M2=10):  # 振动升降指标
        LC = REF(self.close, 1);
        AA = ABS(self.high - LC);
        BB = ABS(self.low - LC);
        CC = ABS(self.high - REF(self.low, 1));
        DD = ABS(LC - REF(self.open, 1));
        R = IF((AA > BB) & (AA > CC), AA + BB / 2 + DD / 4, IF((BB > CC) & (BB > AA), BB + AA / 2 + DD / 4, CC + DD / 4));
        X = (self.close - LC + (self.close - self.open) / 2 + LC - REF(self.open, 1));
        SI = 16 * X / R * MAX(AA, BB);
        ASI = SUM(SI, M1);
        ASIT = MA(ASI, M2);
        return ASI, ASIT

    def XSII(self, N=102, M=7):  # 薛斯通道II
        AA = MA((2 * self.close + self.high + self.low) / 4, 5)  # 最新版DMA才支持 2021-12-4
        TD1 = AA * N / 100;
        TD2 = AA * (200 - N) / 100
        CC = ABS((2 * self.close + self.high + self.low) / 4 - MA(self.close, 20)) / MA(self.close, 20)
        DD = DMA(self.close, CC);
        TD3 = (1 + M / 100) * DD;
        TD4 = (1 - M / 100) * DD
        return TD1, TD2, TD3, TD4

    def SAR(self, N=10, S=2, M=20):
        """
        求抛物转向。 例如SAR(10,2,20)表示计算10日抛物转向，步长为2%，步长极限为20%
        Created by: jqz1226, 2021-11-24首次发表于聚宽(www.joinquant.com)

        :param self.high: self.high序列
        :param self.low: self.low序列
        :param N: 计算周期
        :param S: 步长
        :param M: 步长极限
        :return: 抛物转向
        """
        f_step = S / 100;
        f_max = M / 100;
        af = 0.0
        is_long = self.high[N - 1] > self.high[N - 2]
        b_first = True
        length = len(self.high)

        s_hhv = REF(HHV(self.high, N), 1)  # type: np.ndarray
        s_llv = REF(LLV(self.low, N), 1)  # type: np.ndarray
        sar_x = np.repeat(np.nan, length)  # type: np.ndarray
        for i in range(N, length):
            if b_first:  # 第一步
                af = f_step
                sar_x[i] = s_llv[i] if is_long else s_hhv[i]
                b_first = False
            else:  # 继续多 或者 空
                ep = s_hhv[i] if is_long else s_llv[i]  # 极值
                if (is_long and self.high[i] > ep) or ((not is_long) and self.low[i] < ep):  # 顺势：多创新高 或者 空创新低
                    af = min(af + f_step, f_max)
                #
                sar_x[i] = sar_x[i - 1] + af * (ep - sar_x[i - 1])

            if (is_long and self.low[i] < sar_x[i]) or ((not is_long) and self.high[i] > sar_x[i]):  # 反空 或者 反多
                is_long = not is_long
                b_first = True
        return sar_x


    def TDX_SAR(self, iAFStep=2, iAFLimit=20):  # type: (np.ndarray, np.ndarray, int, int) -> np.ndarray
        """  通达信SAR算法,和通达信SAR对比完全一致   by: jqz1226, 2021-12-18
        :param self.high: 最高价序列
        :param self.low: 最低价序列
        :param iAFStep: AF步长
        :param iAFLimit: AF极限值
        :return: SAR序列
        """
        af_step = iAFStep / 100;
        af_limit = iAFLimit / 100
        SarX = np.zeros(len(self.high))  # 初始化返回数组

        # 第一个bar
        bull = True
        af = af_step
        ep = self.high[0]
        SarX[0] = self.low[0]
        # 第2个bar及其以后
        for i in range(1, len(self.high)):
            # 1.更新：hv, lv, af, ep
            if bull:  # 多
                if self.high[i] > ep:  # 创新高
                    ep = self.high[i]
                    af = min(af + af_step, af_limit)
            else:  # 空
                if self.low[i] < ep:  # 创新低
                    ep = self.low[i]
                    af = min(af + af_step, af_limit)
            # 2.计算SarX
            SarX[i] = SarX[i - 1] + af * (ep - SarX[i - 1])

            # 3.修正SarX
            if bull:
                SarX[i] = max(SarX[i - 1], min(SarX[i], self.low[i], self.low[i - 1]))
            else:
                SarX[i] = min(SarX[i - 1], max(SarX[i], self.high[i], self.high[i - 1]))

            # 4. 判断是否：向下跌破，向上突破
            if bull:  # 多
                if self.low[i] < SarX[i]:  # 向下跌破，转空
                    bull = False
                    tmp_SarX = ep  # 上阶段的最高点
                    ep = self.low[i]
                    af = af_step
                    if self.high[i - 1] == tmp_SarX:  # 紧邻即最高点
                        SarX[i] = tmp_SarX
                    else:
                        SarX[i] = tmp_SarX + af * (ep - tmp_SarX)
            else:  # 空
                if self.high[i] > SarX[i]:  # 向上突破, 转多
                    bull = True
                    ep = self.high[i]
                    af = af_step
                    SarX[i] = min(self.low[i], self.low[i - 1])
        # end for
        return SarX
