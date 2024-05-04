from config import IR_threshold, gp_start, gp_end, gp_predPeriod, gp_dataDir_holc, \
gp_dataDir_h5, gp_us191, gp_us101, gp_pickalphaDir
import sys
sys.path.append(gp_pickalphaDir)
import os
import numpy as np
import pandas as pd
from pick_alpha import alphaTest
from utilities import function_set, feature_names, transform_label, stdization
from cubicalGenetic import genetic, fitness, functions
from calculate_alpha import calculateAlpha


def genBetterFormula(fitResult, threshold):
    '''
    输入fit后的est_gp和fitness的阈值，从最后一代选出超过阈值的公式
    '''
    if fitResult.metric.greater_is_better:
        better_fitness = [ f.__str__() for f in fitResult._programs[-1] if f.fitness_ > threshold] 
    else:
        better_fitness = [ f.__str__() for f in fitResult._programs[-1] if f.fitness_ < threshold]
    return better_fitness

def IR(y, y_pred, w):
    alp = pd.DataFrame(y_pred, index=alphaFitter.univ, columns=alphaFitter.dateRange)
    coef, _, _, _, _ = alphaFitter.singleAlpRegression(alp)
    if coef.isnull().sum() / len(coef) > 0.8 or coef.std() == 0 :
        return 0
    else:
        IR = coef.mean() / coef.std() * np.sqrt(252 / gp_predPeriod)
        if np.isnan(IR):
            return 0
        else:
            return abs(IR)

def ICIR(y, y_pred, w):
    alp = pd.DataFrame(y_pred, index=alphaFitter.univ, columns=alphaFitter.dateRange)
    _, _, _, _, IC = alphaFitter.singleAlpRegression(alp)
    ICIR = IC.mean() / IC.std()
    if np.isnan(ICIR):
        return 0
    else:
        return abs(ICIR)


if __name__ == '__main__':

    alphaFitter = alphaTest(startDate=gp_start, endDate=gp_end, predPeriod=gp_predPeriod, dataDir=gp_dataDir_h5)

    #################################
    openPx  = pd.read_csv(os.path.join(gp_dataDir_holc,'openprice_adj.csv'), index_col=0).reindex(index=alphaFitter.univ, columns=alphaFitter.dateRange)
    highPx  = pd.read_csv(os.path.join(gp_dataDir_holc,'highprice_adj.csv'), index_col=0).reindex(index=alphaFitter.univ, columns=alphaFitter.dateRange)
    lowPx   = pd.read_csv(os.path.join(gp_dataDir_holc,'lowprice_adj.csv'), index_col=0).reindex(index=alphaFitter.univ, columns=alphaFitter.dateRange)
    closePx = pd.read_csv(os.path.join(gp_dataDir_holc,'closeprice_adj.csv'), index_col=0).reindex(index=alphaFitter.univ, columns=alphaFitter.dateRange)
    volume  = pd.read_csv(os.path.join(gp_dataDir_holc,'volume.csv'), index_col=0).reindex(index=alphaFitter.univ, columns=alphaFitter.dateRange)
    vwap    = pd.read_csv(os.path.join(gp_dataDir_holc,'vwap.csv'), index_col=0).reindex(index=alphaFitter.univ, columns=alphaFitter.dateRange)
    industry = pd.read_csv(os.path.join(gp_dataDir_holc,'industry.csv'), index_col=0)
    functions.ind_label = transform_label(industry).reindex(index=alphaFitter.univ)
    
    dataArray = np.concatenate([
        np.expand_dims(openPx.fillna(0), axis=2), 
        np.expand_dims(highPx.fillna(0), axis=2), 
        np.expand_dims(lowPx.fillna(0), axis=2), 
        np.expand_dims(closePx.fillna(0), axis=2), 
        np.expand_dims(volume.fillna(0), axis=2), 
        np.expand_dims(vwap.fillna(0), axis=2), 
    ], axis=2)
    
    ################ UPDATE DATA#################

    est_gp = genetic.SymbolicRegressor(
        population_size = 30, 
        generations = 2, 
        tournament_size = 10, 
        init_depth = (1, 3), 
        stopping_criteria = 500, 
        max_samples = 0.9, 
        low_memory = True, 
        feature_names = feature_names, 
        function_set = function_set, 
        metric = fitness.make_fitness(IR, True), 
        verbose = 1,
        const_range = (3,7),
        init_method = 'grow'
    )
    
    #gp_us101, _ = stdization(gp_us101)
    #gp_us191, _ = stdization(gp_us191)
    #gp_us = gp_us101 + gp_us191
    # est_gp.fit(X=dataArray, y=np.squeeze(alphaFitter.regDataset['regRet']), us=gp_us)
    est_gp.fit(X=dataArray, y=np.squeeze(alphaFitter.regDataset['regRet']))
    #获取符合要求的goodAlpha
    alpha = genBetterFormula(est_gp, IR_threshold)
    # 去重
    alpha = list(set(alpha))
    #完成alpha更新
    calculateAlpha(alpha=alpha, factor_value=False)
    

