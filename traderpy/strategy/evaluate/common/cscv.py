from dataclasses import dataclass
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot
import statsmodels.api as sm
import scipy.stats as ss
from scipy.stats import norm
from scipy.stats import skew as stats_skew
from scipy.stats import kurtosis as stats_kurtosis
from yahooquery import Ticker

jtplot.style('chesterish')


@dataclass
class CSCV:
    """
    Implements the Combinatorially Symmetric Cross-Validation procedure based on the following source:
        - M. L. de Prado, Advances in financial machine learning (John Wiley & Sons, 2018).
    """

    dataframe: pd.DataFrame
    n_sub_matrices: int

    def generate_indices(self):
        """
        Generates indices along the DataFrame that contains the returns of each backtest simulation according to the
        chosen number of submatrices.
        """

        return np.array(
            sum([[i] * (len(self.dataframe) // self.n_sub_matrices) for i in range(self.n_sub_matrices)], []))

    def split(self):
        """
        Partitions the returns DataFrame (across rows) into an even number of disjoint submatrices of equal dimensions
        and forms a train and a test set (also of equal size) combining the submatrices.
        """

        indices = self.generate_indices()
        comb = list(combinations(range(self.n_sub_matrices), int(self.n_sub_matrices / 2)))
        all_splits = range(self.n_sub_matrices)

        for combination in comb:
            train_indices, test_indices = [], []
            for c in combination:
                indices_train = list(np.where(indices == c)[0])
                train_indices.extend(indices_train)
            for t in list(set(all_splits) - set(combination)):
                indices_test = list(np.where(indices == t)[0])
                test_indices.extend(indices_test)
            yield train_indices, test_indices

    def get_n_splits(self):
        """
        According to the chosen number of submatrices, the total number of combinations splits between train and test
        sets is calculated.

        :return: (int): Total number of combinations splits between train and test sets.
        """
        comb = combinations(range(self.n_sub_matrices), int(self.n_sub_matrices / 2))
        return len(list(comb))


class PBO:
    """
    Class that analyzes the overall difference in performance between in sample and out of sample performance through
    several metrics and visualizations, specially the probability of backtest overfitting, following the sources:
        - M. L. de Prado, Advances in financial machine learning (John Wiley & Sons, 2018).
        - D. Bailey, J. Borwein, M. López de Prado and J. Zhu, “The probability of backtest overfitting,” working
          paper, 2013, http://ssrn.com/abstract=2326253.
        - Bailey, D., J. Borwein, M. López de Prado and J. Zhu. “Pseudo-Mathematics and Financial Charlatanism: The
          Effects of Backtest Overfitting on Out-Of-Sample Performance.” Notices of the American Mathematical Society,
          Vol. 61, No. 5 (2014), pp. 458-471. Available at http://ssrn.com/abstract=2308659.
    """

    def __init__(self, data, n_sub_matrices=16):
        """
        Inputs the class attributes.

        :param data: (pd.DataFrame): Matrix that contains the backtests simulations performance measures as columns.
        :param n_sub_matrices: (int, must be an even number): Number of disjoint submatrices of equal dimensions formed
        from the partition of the performance matrix.
        """

        self.data = data
        self.n_sub_matrices = n_sub_matrices
        self.trials_is = []
        self.trials_oos = []
        self.n_star_is = []
        self.relative_star_oos = []
        self.logits = []

    def calculate_pbo(self):
        """
        Estimates the probability of backtest overfitting (PBO) through Combinatorially Symmetric Cross-Validation.
        The PBO can be defined as a non-parametric, model-free procedure that aims to quantify the likelihood of
        backtest overfitting, more specifically, evaluates the conditional probability that a particular strategy
        underperforms the median out of sample set while still optimal in sample.
        """

        cscv = CSCV(self.data, self.n_sub_matrices)

        for i, (train_ids, test_ids) in enumerate(cscv.split(), start=1):
            # print('processing split', i, '/', cscv.get_n_splits())
            # print(len(train_ids), len(test_ids))
            dataset_train, dataset_test = self.data.iloc[train_ids], self.data.iloc[test_ids]
            sharpe_train = dataset_train.mean() / dataset_train.std()
            sharpe_test = dataset_test.mean() / dataset_test.std()
            n_star = np.argmax(sharpe_train)
            rank_sharpe_test = sharpe_test.rank()
            relative_rank = (rank_sharpe_test[n_star]) / (len(self.data.columns))
            logit = np.log(relative_rank / (1 - relative_rank))
            self.logits.append(logit)
            for j in range(len(self.logits)):
                if self.logits[j] == -np.inf:
                    self.logits[j] = 0
                elif self.logits[j] == np.inf:
                    self.logits[j] = 0
            self.trials_is.append(sharpe_train)
            self.trials_oos.append(sharpe_test)
            self.n_star_is.append(sharpe_train[n_star])
            self.relative_star_oos.append(sharpe_test[n_star])

        # Probability of Backtest Overfitting
        logits_array = np.array(self.logits)
        negative_logits = logits_array[logits_array < 0]
        pbo = len(negative_logits) / len(logits_array)
        plt.figure(figsize=(15, 4))
        sns.distplot(logits_array, label='PBO = ' + str(np.round(pbo, 2)))
        plt.title('Histogram of Rank Logits', fontsize=20)
        plt.grid(False)
        plt.legend()
        plt.show()

    def performance_degradation(self):
        """
        Performs a linear regression between the best strategy's simulations performances in sample and the out of
        sample performance for the same simulation, aiming to evaluate de degree of overall performance degradation
        of the trading strategy.
        """

        n_star_is = pd.DataFrame(self.n_star_is)
        relative_star_oos = pd.DataFrame(self.relative_star_oos)
        X, Y = n_star_is, relative_star_oos
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()

        x1 = np.linspace(np.min(n_star_is), np.max(n_star_is), 20)
        y1 = results.params['const'] + results.params[0] * x1
        plt.figure(figsize=(15, 4))
        plt.scatter(n_star_is, relative_star_oos)
        plt.plot(x1, y1, color='red', ls='--', label=f'Regression slope: {str(np.round(results.params[0], 2))} ; '
                                                        f'P-value: {str(np.round(results.f_pvalue, 2))}')
        plt.title('Out of Sample Performance Degradation', fontsize=20)
        plt.xlabel('Performance IS')
        plt.ylabel('Performance OOS')
        plt.grid(False)
        plt.legend()
        plt.show()

    def is_oos_performance(self):
        """
        Visualization of in sample and out of sample performances distributions.
        """

        n_star_is = pd.DataFrame(self.n_star_is)
        relative_star_oos = pd.DataFrame(self.relative_star_oos)
        plt.figure(figsize=(15, 4))
        plt.title('Distributions of IS and OOS Statistics', fontsize=20)
        sns.distplot(n_star_is, label='In Sample', color='black', kde=True)
        sns.distplot(relative_star_oos, label='Out of Sample', color='lightskyblue', kde=True)
        plt.grid(False)
        plt.legend()
        plt.show()


class SharpeAnalysis:
    """
    For each backtest simulation, provides methods for calculating the probabilistic Sharpe ratio, the minimum track
    record length and the deflated Sharpe ratio, and further plots the distribution of the deflated Sharpes of the
    strategy's trials.
    Sources:
        - M. L. de Prado, Advances in financial machine learning (John Wiley & Sons, 2018).
        - M. L. de Prado. Machine Learning for Asset Managers. Cambridge University Press, 2020.
        - Bailey, D. and M. López de Prado. “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
          Overfitting and Non-Normality.” Journal of Portfolio Management, Vol. 40, No. 5 (2014), pp. 94-107.
        - Bailey, David; de Prado, Marcos, ” The Sharpe ratio efficient frontier”, The Journal of Risk, Winter
          2012/2013, Vol.15(2), pp. 3-44.
        - Lopez de Prado, Marcos. "Deflating the Sharpe Ratio." Available at SSRN 2465675 (2014).
    """

    def __init__(self, trials_pnls, start_test, end_test, obs_sharpes, obs_n_trades, ref_sr=1., conf_interval=0.95):
        """
        Initializes the class attributes.

        :param trials_pnls: (pd.DataFrame): Table with the strategy's trials as columns and their respective net result
        (profits or losses) of each trade performed during the backtest period
        :param start_test: (str): Starting date of the test period
        :param end_test: (str): Final date of the test period
        :param obs_sharpes: (list): Sharpe ratio of every backtest simulation
        :param obs_n_trades: (list): Number of trades of every backtest simulation
        :param ref_sr: (float): Benchmark Sharpe ratio used as reference for calculating the probabilistic Sharpe ratio
        :param conf_interval: (float): Confidence interval for accessing the statistical significance of a particular
        estimate.
        """
        self.trials_pnls = trials_pnls
        self.start_test = start_test
        self.end_test = end_test
        self.obs_sharpes = obs_sharpes
        self.obs_n_trades = obs_n_trades
        self.ref_sr = ref_sr
        self.conf_interval = conf_interval

    def psr(self, obs_sr, skew, kurtosis, number_trades):
        """
        Calculates the probabilist Sharpe ratio:
            - Corrects the Sharpe ratio for the inflationary effects of Non-Normal returns and track record length.
            - Designed for single trial testing.
            - It should exceed 0.95 for the standard significance level of 5%.

        :param obs_sr: (float): Sharpe ratio estimate of a particular strategy trial
        :param skew: (float): Skew of a particular trial's performance measure
        :param kurtosis: (float): Kurtosis of a particular trial's performance measure
        :param number_trades: (int): Number of trades of a given backtest simulation
        :return: (float): Probabilistic Sharpe ratio estimate.
        """

        probabilistic_sr = norm.cdf((obs_sr - self.ref_sr) * (number_trades - 1) ** 0.5 /
                                    (1 - obs_sr * skew + obs_sr ** 2 * (kurtosis - 1) / 4.) ** 0.5)
        return f'Probabilistic Sharpe Ratio {probabilistic_sr:2.2f}'

    def min_trl(self, obs_sr, skew, kurtosis):
        """
        Estimates how long the track record of the strategy should be in order to achieve statistical confidence that
        its Sharpe ratio is above a particular threshold.

        :param obs_sr: (float): Sharpe ratio estimate of a particular strategy trial
        :param skew: (float): Skew of a particular trial's performance measure
        :param kurtosis: (float): Kurtosis of a particular trial's performance measure
        :return: (float): Track record length estimate
        """

        minimum_trl = 1 + (1 - skew * obs_sr + (kurtosis - 1) / 4. * obs_sr ** 2) * (
                norm.ppf(self.conf_interval) / (obs_sr - self.ref_sr)) ** 2
        return minimum_trl

    @staticmethod
    def dsr(obs_sr, skew, kurtosis, variance_sharpes, strategy_duration, number_trials):
        """
        Calculates the deflated Sharpe ratio:
            - It can be defined as a probabilistic Sharpe ratio with the difference that the rejection threshold is
              adjusted for accounting the multiplicity of independent trials.

        :param obs_sr: (float): Sharpe ratio estimate of a particular strategy trial
        :param skew: (float): Skew of a particular trial's performance measure
        :param kurtosis: (float): Kurtosis of a particular trial's performance measure
        :param variance_sharpes: (float): Variance measure of the trails' Sharpe ratios
        :param strategy_duration: (float):Duration in days of the strategy
        :param number_trials: (float): Estimate of the number of independent trials of the strategy
        :return: (float): Deflated Sharpe ratio estimate
        """

        EM = 0.577215664901533  # Euler-Mascheroni constant
        dsr_reference = np.sqrt(variance_sharpes) * (((1 - EM) * norm.ppf(
            1 - 1 / number_trials)) + EM * norm.ppf(1 - 1 / (number_trials * np.e)))
        deflated_sr = norm.cdf((obs_sr - dsr_reference) * (strategy_duration - 1) ** 0.5 / np.sqrt(
            1 - obs_sr * skew + obs_sr ** 2 * (kurtosis - 1) / 4.))
        return deflated_sr

    def distribution_dsr(self):
        """
        Plots the empirical distribution of backtests simulations' deflated Sharpe ratios.
        """

        sharpes = np.array(self.obs_sharpes / np.sqrt(252))
        var_sharpe = np.array(self.obs_sharpes).var() * (1 / 252)
        strategy_duration = np.busday_count(self.start_test.date(), self.end_test.date())
        corr_matrix = self.trials_pnls.corr()
        trials_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
        independent_trials = (trials_corr + (1 - trials_corr)) * self.trials_pnls.shape[1]
        sharpes_list = []
        for i, j in zip(sharpes, self.trials_pnls):
            pnls = self.trials_pnls[j].dropna()
            skew = stats_skew(pnls)
            kurtosis = stats_kurtosis(pnls)
            sharpe = self.dsr(i, skew, kurtosis, var_sharpe, strategy_duration, independent_trials)
            sharpes_list.append(sharpe)
        sharpes_array = np.array(sharpes_list)
        st_significance_sharpes = sharpes_array[sharpes_array >= self.conf_interval]
        proportion_st_significance = len(st_significance_sharpes) / len(sharpes_array)

        # Plot distribution of deflated Sharpe ratios
        plt.figure(figsize=(15, 4))
        sns.distplot(sharpes_list, label='Deflated Sharpe ratios')
        plt.title('Distribution of Deflated Sharpe Ratios', fontsize=20)
        plt.grid(False)
        plt.legend()
        plt.show()
        print(f'Proportion of deflated Sharpe ratios below {1 - self.conf_interval:2.2f} '
              f'statistical significance level: {proportion_st_significance:2.2f}')


class StrategyRisk:
    """
    For each backtest simulation and a given set of defined parameters, implements methods for calculating the following
    metrics:
        - Minimum strategy's precision rate
        - Minimum Sharpe ratio target
        - Number of bets per year for achieving predefined Sharpe ratio target
        - Probability of strategy failure
    In addition, the following distributions are provided:
        - Percentage of positive trades of the strategy's trials
        - Probability of failure of the strategy's trials
    Sources:
        - M. L. de Prado, Advances in financial machine learning (John Wiley & Sons, 2018).
    """

    def __init__(self, obs_n_trades, start_test, end_test, trials_pnls, target_sr=1.):
        """
        Initializes the class attributes.

        :param obs_n_trades: (int): Number of trades of a given backtest simulation
        :param start_test: (str): Starting date of the test period
        :param end_test: (str): Final date of the test period
        :param trials_pnls: (pd.DataFrame): Table with the strategy's trials as columns and their respective net result
        (profits or losses) of each trade performed during the backtest period
        :param target_sr: Target Sharpe ratio set as the investor's objective
        """

        self.obs_n_trades = obs_n_trades
        self.start_test = start_test
        self.end_test = end_test
        self.trials_pnls = trials_pnls
        self.target_sr = target_sr

    def min_p(self, sl, pt, frequency):
        """
        For a given trading strategy with predefined stop loss, take profit and betting frequency (per year) parameters,
        it calculates the required precision rate for achieving a target Sharpe ratio.

        :param sl: (float): Stop-loss of the strategy
        :param pt: (float): Take-profit of the strategy
        :param frequency: (float): Early betting frequency of the strategy
        :return: (float): Minimum precision rate estimate
        """

        a = (frequency + self.target_sr ** 2) * (pt - sl) ** 2
        b = (2 * frequency * sl - self.target_sr ** 2 * (pt - sl)) * (pt - sl)
        c = frequency * sl ** 2
        minimum_p = (-b + (b ** 2 - 4 * a * c) ** .5) / (2. * a)
        return minimum_p

    @staticmethod
    def ann_sr(sl, pt, p, frequency):
        """
        For a given trading strategy with predefined stop loss, take profit, betting frequency (per year) and precision
        rate parameters, the annualized Sharpe ratio estimate of the trading strategy is calculated.

        :param sl: (float): Stop-loss of the strategy
        :param pt: (float): Take-profit of the strategy
        :param p: (float): Precision rate of the strategy
        :param frequency: (float): Early betting frequency of the strategy
        :return: Annualized Sharpe ratio estimate of the strategy
        """

        return ((pt - sl) * p + sl) / ((pt - sl) * (p * (1 - p)) ** .5) * frequency ** .5

    def min_freq(self, sl, pt, p, frequency):
        """
        For a given trading strategy with predefined stop loss, take profit, betting frequency (per year) and precision
        rate parameters, the minimum number of trading bets per year for achieving a particular Sharpe ratio is
        calculated.

        :param sl: (float): Stop-loss of the strategy
        :param pt: (float): Take-profit of the strategy
        :param p: (float): Precision rate of the strategy
        :param frequency: (float): Early betting frequency of the strategy
        :return: (float): Minimum number of trading bets per year for achieving a target Sharpe ratio
        """

        minimum_freq = (self.target_sr * (pt - sl)) ** 2 * p * (1 - p) / ((pt - sl) * p + sl) ** 2
        if not np.isclose(self.ann_sr(sl, pt, p, frequency), self.target_sr):
            return
        return minimum_freq

    def positive_trades(self):
        """
        Plots the empirical distribution of percentage of positive trades of the strategy trials.
        """

        proportion_list = []
        for i in self.trials_pnls:
            trades = self.trials_pnls[i].dropna()
            number_trades = len(trades)
            positive = len(trades[trades > 0])
            proportion = positive / number_trades
            proportion_list.append(proportion)
        proportion_array = np.array(proportion_list)
        above_half = proportion_array[proportion_array > 0.50]
        proportion_above_half = len(above_half) / len(proportion_array)

        plt.figure(figsize=(15, 4))
        sns.distplot(proportion_list)
        plt.title("Distribution of Trails' Positive Trades Fraction", fontsize=20)
        plt.grid(False)
        plt.show()
        print(f'Trials with percentage of positive trades above 0.50: {proportion_above_half:2.2f}')

    def probability_failure(self, trial_pnls, frequency):
        """
        Estimates the probability that the strategy will fail.

        :param trial_pnls: (pd.Series): Net result (profit or loss) of each trade performed during the backtest period
        of the strategy's trial
        :param frequency: (float): Early betting frequency of the strategy's trial
        :return: Strategy's probability of failure
        """

        trial_pnls.dropna()
        r_pos = trial_pnls[trial_pnls > 0].mean()
        r_neg = trial_pnls[trial_pnls <= 0].mean()
        empirical_p = len(trial_pnls[trial_pnls > 0]) / len(trial_pnls)
        p_threshold = self.min_p(r_neg, r_pos, frequency)
        risk = ss.norm.cdf(p_threshold, empirical_p, empirical_p * (1 - empirical_p))  # approximation to bootstrap
        return risk

    def distribution_prob_f(self):
        """
        Plots the empirical distribution of probability of failures of the strategy trials.
        """

        strategy_duration = np.busday_count(self.start_test.date(), self.end_test.date())
        trades_per_year = [i / strategy_duration * 252 for i in self.obs_n_trades]
        frequencies_array = np.array(trades_per_year)
        prob_list = []
        for i, j in zip(self.trials_pnls, frequencies_array):
            pnls = self.trials_pnls[i].dropna()
            probability = self.probability_failure(pnls, j)
            prob_list.append(probability)
        prob_array = np.array(prob_list)
        prob_below = prob_array[prob_array <= 0.05]
        proportion_prob_below = len(prob_below) / len(prob_array)

        plt.figure(figsize=(15, 4))
        sns.distplot(prob_list)
        plt.title('Distribution of Probabilities of Failure', fontsize=20)
        plt.grid(False)
        plt.show()
        print(f'Proportion of trials with probability of failure below 0.05: {proportion_prob_below:2.2f}')


class VisualizeEquity:
    """
    Class that outputs visualizations of the equity curve of the strategy trials and compares it with the Bova11 ETF,
    which tracks the Ibovespa (most popular brazilian market index composed with the most traded stocks in Brazil).
    """

    def __init__(self, equity, returns, start_test, end_test, interval='5m'):
        """
        Initializes the class attributes.

        :param equity: (pd.DataFrame): Portfolio market value of every backtest simulation at every point in time.
        :param returns: (pd.DataFrame): Returns of every backtest simulation at every point in time.
        :param start_test: (str): Starting date of the test period
        :param end_test: (str): Final date of the test period
        :param interval: (str): Time interval between data points
        """

        self.equity = equity
        self.returns = returns
        self.start_test = start_test
        self.end_test = end_test
        self.interval = interval

    def equity_curve(self):
        """
        Plots the evolving equity market value of the backtests simulations and the mean performance of all trials.
        """

        equity_diff = self.equity.diff()
        equity_curve = equity_diff.cumsum()
        plt.figure(figsize=(16, 6))
        plt.plot(equity_curve, color='grey', ls='--', lw=0.5)
        plt.title('Equity Curve', fontsize=20)
        plt.plot(equity_curve.mean(axis=1), lw=3, label='Mean Performance')
        plt.grid(False)
        plt.legend()
        plt.show()

    def equity_returns(self):
        """
        Plots the evolving equity returns of the backtests simulations, the mean performance of all trials and the
        returns of the Bova11 ETF for the same period.
        """

        ibov = Ticker('BOVA11.SA')
        bova11 = ibov.history(start=self.start_test, end=self.end_test, interval=self.interval)['close']
        ret_bova11 = bova11.pct_change()
        ret_bova11.fillna(0)
        benchmark = np.array((1 + ret_bova11).cumprod() - 1)
        cum_returns = (1 + self.returns).cumprod() - 1
        plt.figure(figsize=(16, 6))
        plt.title('Strategy x BOVA11', fontsize=20)
        for trial in cum_returns:
            plt.plot(cum_returns['{}'.format(trial)], color='grey', ls='--', lw=0.5)
        plt.plot(cum_returns.mean(axis=1), lw=3, label='Mean Performance')
        plt.plot(benchmark, lw=3, label='BOVA11', color='lightskyblue')
        plt.grid(False)
        plt.legend()
        plt.show()
