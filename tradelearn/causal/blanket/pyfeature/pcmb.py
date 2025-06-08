import numpy as np
from itertools import combinations
from .condition_independence_test import cond_indep_test


def subsets(nbrs, k):
    return set(combinations(nbrs, k))

def getPCD(data, target, alaph, is_discrete):
    number, kVar = np.shape(data)
    max_k = 3
    PCD = []
    ci_number = 0

    # use a list of sepset[] to store a condition set which can make target and the variable condition independence
    # the above-mentioned variable will be remove from CanPCD or PCD
    sepset = [[] for i in range(kVar)]

    while True:
        variDepSet = []
        CanPCD = [i for i in range(kVar) if i != target and i not in PCD]
        CanPCD_temp = CanPCD.copy()

        for vari in CanPCD_temp:
            breakFlag = False
            dep_gp_min = float("inf")
            vari_min = -1

            if len(PCD) >= max_k:
                Plength = max_k
            else:
                Plength = len(PCD)

            for j in range(Plength+1):
                SSubsets = subsets(PCD, j) ## {{1，2，4} {13} {23}}
                for S in SSubsets:
                    ci_number += 1
                    pval_gp, dep_gp = cond_indep_test(data, target, vari, S, is_discrete)

                    if pval_gp > alaph:
                        vari_min = -1
                        CanPCD.remove(vari)
                        sepset[vari] = [i for i in S] # 出现独立的则记录sepset
                        breakFlag = True
                        break
                    elif dep_gp < dep_gp_min:
                        dep_gp_min = dep_gp
                        vari_min = vari

                if breakFlag:
                    break

            # use a list of variDepset to store list, like [variable, its dep]
            if vari_min in CanPCD:
                variDepSet.append([vari_min, dep_gp_min]) # 最可能依赖的项，一个变量一个对应项，找到sepset则-1，否则自己

        # sort list of variDepSet by dep from max to min    根据统计量由大到小排序
        variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)

        # if variDepset is null ,that meaning PCD will not change
        if variDepSet != []:  ### --- 这里还好理解
            y =variDepSet[0][0] ### 找到最可能的Y
            PCD.append(y)
            pcd_index = len(PCD)
            breakALLflag = False
            while pcd_index >=0: ### 遍历每一个X
                pcd_index -= 1
                x = PCD[pcd_index]
                breakFlagTwo = False

                conditionSetALL = [i for i in PCD if i != x]
                if len(conditionSetALL) >= max_k:
                    Slength = max_k
                else:
                    Slength = len(conditionSetALL) # 这儿也设置了长度限制

                for j in range(Slength+1):
                    SSubsets = subsets(conditionSetALL, j)
                    for S in SSubsets:
                        ci_number += 1
                        pval_sp, dep_sp = cond_indep_test(data, target, x, S, is_discrete)

                        if pval_sp > alaph:

                            PCD.remove(x)
                            if x == y: ### 如果第一个就独立，则提前结束
                                breakALLflag = True

                            sepset[x] = [i for i in S]
                            breakFlagTwo = True ### 完成一个变量的检测
                            break
                    if breakFlagTwo:
                        break

                if breakALLflag:
                    break
        else:
            break
    return list(set(PCD)), sepset, ci_number


def getPC(data, target, alaph, is_discrete):
    ci_number = 0
    PC = []
    PCD, sepset, ci_num2 = getPCD(data, target, alaph, is_discrete)
    ci_number += ci_num2
    for x in PCD:
        variSet, _, ci_num3 = getPCD(data, x, alaph, is_discrete)
        ci_number += ci_num3
        # PC of target ,whose PC also has the target, must be True PC
        if target in variSet:
            PC.append(x)

    return list(set(PC)), sepset, ci_number


def PCMB(data, target, alaph, is_discrete=True):
    PC, sepset, _ = getPC(data, target, alaph, is_discrete)

    MB = PC.copy()

    for x in PC:
        PCofPC_temp, _, _ = getPC(data, x, alaph, is_discrete)

        PCofPC = [i for i in PCofPC_temp if i != target and i not in MB]

        for y in PCofPC:
            conditionSet = [i for i in sepset[y]]
            conditionSet.append(x)
            conditionSet = list(set(conditionSet))

            pval, dep = cond_indep_test(
                data, target, y, conditionSet, is_discrete)
            if pval <= alaph:
                MB.append(y)
                break
    return list(set(MB))

