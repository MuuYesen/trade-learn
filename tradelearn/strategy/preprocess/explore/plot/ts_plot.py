import os
cur_dir_path = os.path.abspath(os.path.dirname(__file__))
import json
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class TsPlot:
    '''
    The following class use matplotlib to plot the data from pandas dataframe. 
    It is made to be simple and easy to use and can be used in the jupyter notebook.
    The main focus is to plot time series data and to use as first tool to visualize the data.
    '''
    def __init__(self, df:pd.DataFrame, date_column:str, setting_dict: dict=None, theme_name:str='default') -> None:
        if (date_column not in df.columns)&(date_column not in df.index.names):
            raise ValueError(f"{date_column} not found in DataFrame columns")
        self.data = df
        self.date_column = date_column
        self.setting_dict = setting_dict
        self.theme_name = theme_name

        # check that date_column in the index
        if self.date_column not in self.data.index.names:
            self.data.set_index(self.date_column, inplace=True)
        
        if self.theme_name:
            self._apply_theme(self.theme_name)

    # Apply theme
    def _apply_theme(self, theme_name=None):
        '''
        Apply the theme to the plot.
        The theme is a dictionary that contains the plot settings.
        The different options are:
            - 'default': The default theme.
            - 'dark': A dark theme.
            - 'light': A light theme.
            - 'custom': A custom theme.
        '''
        self.theme_name = theme_name if theme_name else self.theme_name

        with open(os.path.join(cur_dir_path, f'default_theme.json')) as json_file:
            theme = json.load(json_file)

        plt.rcParams.update({
            'figure.figsize': theme['figure']['figsize'],
            'figure.dpi': theme['figure']['dpi'],
            'figure.facecolor': theme['figure']['facecolor'],
            'axes.grid': theme['axes']['grid'],
            'axes.labelpad': theme['axes']['labelpad'],
            'axes.labelsize': theme['axes']['labelsize'],
            'axes.titlesize': theme['axes']['titlesize'],
            'axes.titlepad': theme['axes']['titlepad'],
            'axes.titleweight': theme['axes']['titleweight'],
            'axes.titlecolor': theme['axes']['titlecolor'],
            'axes.facecolor': theme['axes']['facecolor'],
            'font.family': theme['font']['family'],
            'font.serif': theme['font']['serif'],
            'grid.linewidth': theme['grid']['linewidth'],
            'grid.color': theme['grid']['color'],
            'xtick.labelsize': theme['ticks']['xtick']['labelsize'],
            'xtick.major.pad': theme['ticks']['xtick']['major_pad'],
            'xtick.color': theme['ticks']['xtick']['color'],
            'ytick.labelsize': theme['ticks']['ytick']['labelsize'],
            'ytick.major.pad': theme['ticks']['ytick']['major_pad'],
            'ytick.color': theme['ticks']['ytick']['color'],
            'legend.fontsize': theme['legend']['fontsize'],
            'legend.facecolor': theme['legend']['facecolor'],
            'legend.framealpha': theme['legend']['framealpha'],
            'legend.edgecolor': theme['legend']['edgecolor'],
            'legend.title_fontsize': theme['legend']['title_fontsize'],
            'legend.shadow': theme['legend']['shadow'],
            'legend.fancybox': theme['legend']['fancybox'],
            'lines.linewidth': theme['lines']['linewidth'],
            'text.color': theme['text']['color'],
        })

        # Assign colors for different plot types if they are available in the theme
        self.line_colors = theme.get('colors', {}).get('line_colors', [])
        self.bar_colors = theme.get('colors', {}).get('bar_colors', [])
        self.histogram_colors = theme.get('colors', {}).get('histogram_colors', [])

    # Plot Lines plot
    def line_plot(self, ts_column: str, display=True, save: bool = False, save_path: str = None):
        '''
        Plot the time series data as line plot using a color from the custom line color cycle. 
        The x-axis is the date and the y-axis is the ts_column.

        Input:
            ts_column: str
                The column name of the time series data.
            display: bool (optional, default=True)
                If True, the plot will be displayed.
            save: bool (optional, default=False)
                If True, the plot will be saved.
            save_path: str (optional, default=None)
                The path to save the plot.
        Output:
            None
        '''

        # Select color from custom cycle
        color = self.line_colors[0] if self.line_colors else 'blue'

        plt.plot(self.data.index, self.data[ts_column].values, label=ts_column, color=color)
        plt.xlabel('Date')
        plt.ylabel(f'{ts_column}')
        plt.title(f'{ts_column} Over Time')
        plt.tight_layout()

        if save:
            self.save_plot(file_name= f'{ts_column} Over Time',save_path=save_path)

        if display:
            plt.show()
        else:
            plt.close()
                  
    def lines_plot(self, ts_columns: list, display=True, save: bool = False, save_path: str = None):
        '''
        Plot multiple time series data as line plot using colors from the custom line color cycle.
        The x-axis is the date and the y-axis is the values from ts_columns.

        Input:
            ts_columns: list
                The list of column names of the time series data.
            display: bool (optional, default=True)
                If True, the plot will be displayed.
            save: bool (optional, default=False)
                If True, the plot will be saved.
            save_path: str (optional, default=None)
                The path to save the plot.
        Output:
            None
        '''

        for i, col in enumerate(ts_columns):
            # Cycle through custom colors, reset to start if the end of the list is reached
            color = self.line_colors[i % len(self.line_colors)] if self.line_colors else None

            plt.plot(self.data.index, self.data[col].values, label=col, color=color)

        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title(f'Time Series')
        plt.tight_layout()
        plt.legend()

        if save:
            self.save_plot(file_name= 'Time Series',save_path=save_path)

        if display:
            plt.show()
        else:
            plt.close()

    # Statistical Plots
    def autoCorrelation_plot(self,ts_column:str,display=True, save:bool=False, save_path:str=None):
        '''
        Plot the auto correlation of the time series data.
        The x-axis is the lag and the y-axis is the auto correlation.
        
        Input:
            ts_column: str
                The column name of the time series data.
            display: bool (optional, default=True)
                If True, the plot will be displayed.
            save: bool (optional, default=False)
                If True, the plot will be saved.
            save_path: str (optional, default=None)
                The path to save the plot.
        Output:
            None
        '''
        
        # get the auto correlation
        acf_values,conf,_,_ = acf(self.data[ts_column],alpha=0.05,qstat=True)
        
        # get the upper and lower bound
        upper_bound = conf[:, 1] - acf_values
        lower_bound = (acf_values - conf[:, 0])*-1
        
        # get the number of lags
        num_lags = np.arange(len(acf_values))
        
        
        plt.plot(num_lags,lower_bound,linestyle='dashed',color='red')
        plt.bar(num_lags,acf_values)
        plt.plot(num_lags,upper_bound,linestyle='dashed',color='red')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.title(f'{ts_column }: Auto Correlation')
        plt.tight_layout()
        
        if save:
            self.save_plot(file_name= f'{ts_column }: Auto Correlation',save_path=save_path)
            
        if display:
            plt.show()
        else:
            plt.close()

    def partialAutoCorrelation_plot(self,ts_column:str,display=True, save:bool=False, save_path:str=None):
        '''
        Plot the partial auto correlation of the time series data.
        The x-axis is the lag and the y-axis is the partial auto correlation.
        
        Input:
            ts_column: str
                The column name of the time series data.
            display: bool (optional, default=True)
                If True, the plot will be displayed.
            save: bool (optional, default=False)
                If True, the plot will be saved.
            save_path: str (optional, default=None)
                The path to save the plot.
        Output:
            None
        '''
        
        # get the partial auto correlation
        pacf_values,conf= pacf(self.data[ts_column],alpha=0.05,method='ols')
        
        # get the upper and lower bound
        upper_bound = conf[:, 1] - pacf_values
        lower_bound = (pacf_values - conf[:, 0])*-1
        
        # get the number of lags
        num_lags = np.arange(len(pacf_values))

        plt.plot(num_lags,lower_bound,linestyle='dashed',color='red')
        plt.bar(num_lags,pacf_values)
        plt.plot(num_lags,upper_bound,linestyle='dashed',color='red')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.title(f'{ts_column }: Partial Auto Correlation')
        plt.tight_layout()
        
        if save:
            self.save_plot(file_name= f'{ts_column }: Partial Auto Correlation',save_path=save_path)
            
        if display:
            plt.show()
        else:
            plt.close()
                   
    # Plot underline distribution estimate
    def histogram_plot(self,ts_column:str,display=True, save:bool=False, save_path:str=None,bin_size:int=100):
        '''
        Plot the histogram of the time series data.
        The x-axis is the value and the y-axis is the frequency.
        
        Input:
            ts_column: str
                The column name of the time series data.
            display: bool (optional, default=True)
                If True, the plot will be displayed.
            save: bool (optional, default=False)
                If True, the plot will be saved.
            save_path: str (optional, default=None)
                The path to save the plot.
        Output:
            None
        '''
        plt.grid(False)
        plt.hist(self.data[ts_column],bins=bin_size)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'{ts_column }: Histogram')
        plt.tight_layout()
        
        if save:
            self.save_plot(file_name= f'{ts_column }: Histogram',save_path=save_path)
            
        if display:
            plt.show()
        else:
            plt.close() 

    def box_plot(self, ts_column: str, display=True, save: bool = False, save_path: str = None):
        if ts_column not in self.data.columns:
            raise ValueError(f"{ts_column} not found in DataFrame columns")

        self.data[ts_column].plot(kind='box')
        plt.ylabel(f'{ts_column}')
        plt.title(f'{ts_column} Box Plot')
        plt.tight_layout()

        if save:
            self.save_plot(file_name= f'{ts_column} Box Plot',save_path=save_path)

        if display:
            plt.show()
        else:
            plt.close()
    
    def stem_plot(self, ts_column: str, display=True, save: bool = False, save_path: str = None):
        if ts_column not in self.data.columns:
            raise ValueError(f"{ts_column} not found in DataFrame columns")
        
        plt.stem(self.data[ts_column])
        plt.ylabel(f'{ts_column}')
        plt.title(f'{ts_column} Stem Plot')
        plt.tight_layout()

        if save:
            self.save_plot(file_name= f'{ts_column} Stem Plot',save_path=save_path)
            
        if display:
            plt.show()
        else:
            plt.close()
        
    # Plot Full statistics including ACF, PACF, Trend, Seasonality, Residuals
    def full_statistics_plot(self, ts_column: str, display=True, save:bool=False, save_path:str=None):
        '''
        Comprehensive plot showing various statistical aspects of a time series.
        Includes the main time series, ACF, PACF, trend, seasonality, and residuals.
        
        Input:
            ts_column: str
                The column name of the time series data.
            display: bool (optional, default=True)
                If True, the plot will be displayed.
            save: bool (optional, default=False)
                If True, the plot will be saved.
            save_path: str (optional, default=None)
                The path to save the plot.
        Output:
            None
        '''

        # Decompose time series
        decomposition = seasonal_decompose(self.data[ts_column], model='additive', period=3)

        # use line color from theme
        color = self.line_colors[0] if self.line_colors else 'black'
        color2 = self.line_colors[1] if self.line_colors else 'red'
        color3 = self.line_colors[2] if self.line_colors else 'blue'
        
        # Create 5x2 grid layout
        fig = plt.figure(figsize=(30, 18))

        # Main time series plot
        ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2)
        ax1.plot(self.data.index, self.data[ts_column], label='Original', color=color)
        ax1.plot(self.data[ts_column].rolling(window=3).mean(), label='3-Step MA', color=color2, linewidth=1)
        ax1.plot(self.data[ts_column].rolling(window=3).std(), label='3-Step Std', color=color3, linestyle='dashed', linewidth=1)
        ax1.set_title(f'Main Time Series: {ts_column}')
        ax1.legend(loc='upper left')

        # ACF plot with confidence interval
        ax2 = plt.subplot2grid((5, 2), (1, 0))
        acf_values, conf_int = acf(self.data[ts_column], nlags=20, alpha=0.05)
        ax2.bar(range(len(acf_values)), acf_values, width=0.3)
        ax2.fill_between(range(len(acf_values)), conf_int[:, 0] - acf_values, conf_int[:, 1] - acf_values, color='red', alpha=0.2)
        ax2.set_title('Autocorrelation Function')

        # PACF plot with confidence interval
        ax3 = plt.subplot2grid((5, 2), (1, 1))
        pacf_values, conf_int = pacf(self.data[ts_column], nlags=20, alpha=0.05, method='ols')
        ax3.bar(range(len(pacf_values)), pacf_values, width=0.3)
        ax3.fill_between(range(len(pacf_values)),conf_int[:, 0] - pacf_values, conf_int[:, 1] - pacf_values , color='red', alpha=0.2)
        ax3.set_title('Partial Autocorrelation Function')

        # Trend plot with linear regression
        ax4 = plt.subplot2grid((5, 2), (2, 0), colspan=2)
        trend = decomposition.trend.dropna()
        X = np.array(range(len(trend))).reshape(-1, 1)
        Y = trend.values
        model = LinearRegression().fit(X, Y)
        trend_line = model.predict(X)
        r_squared = model.score(X, Y)
        ax4.plot(trend.index, trend, color=color ,label='Trend')
        ax4.plot(trend.index, trend_line, label=f'LR(R^2: {r_squared:.2f})', linestyle='dashed',color=color2)
        ax4.set_title('Trend with Linear Regression')
        ax4.legend(loc='upper left')

        # Seasonality plot
        ax5 = plt.subplot2grid((5, 2), (3, 0), colspan=2)
        ax5.plot(decomposition.seasonal,color=color)
        ax5.set_title('Seasonality')

        # Residuals stem plot
        ax6 = plt.subplot2grid((5, 2), (4, 0), colspan=2)
        ax6.stem(decomposition.resid)
        ax6.set_title('Residuals')

        # Layout adjustments
        plt.tight_layout()
        
        # Save or display the plot
        if save:
            self.save_plot(file_name=f'{ts_column}_Statistics', save_path=save_path)

        if display:
            plt.show()
        else:
            plt.close()
        
        return fig

    # Calculate the correlation between two time series
    def correlation(self, ts_column1: str, ts_column2: str, display=True, save: bool = False, save_path: str = None):
        '''
        Calculate the correlation between two time series.
        The correlation is calculated using the Pearson correlation coefficient.

        Input:
            ts_column1: str
                The column name of the first time series data.
            ts_column2: str
                The column name of the second time series data.
            display: bool (optional, default=True)
                If True, the plot will be displayed.
            save: bool (optional, default=False)
                If True, the plot will be saved.
            save_path: str (optional, default=None)
                The path to save the plot.
        Output:
            correlation: float
                The Pearson correlation coefficient.
        '''
        if ts_column1 not in self.data.columns:
            raise ValueError(f"{ts_column1} not found in DataFrame columns")
        if ts_column2 not in self.data.columns:
            raise ValueError(f"{ts_column2} not found in DataFrame columns")

        correlation = self.data[ts_column1].corr(self.data[ts_column2])
        # use line color from theme
        color = self.line_colors[0] if self.line_colors else 'black'
        
        # plot as scatter
        plt.scatter(self.data[ts_column1],self.data[ts_column2])
        # add regression line
        x = np.array(self.data[ts_column1]).reshape(-1,1)
        y = np.array(self.data[ts_column2]).reshape(-1,1)
        model = LinearRegression().fit(x,y)
        y_pred = model.predict(x)
        plt.plot(x,y_pred,color=color,linestyle='-')
        plt.xlabel(ts_column1)
        plt.ylabel(ts_column2)
        plt.title(f'{ts_column1} vs {ts_column2}: Correlation {correlation:.2f}')
        
        if save:
            self.save_plot(file_name= f'{ts_column1} vs {ts_column2}',save_path=save_path)

        if display:
            print(f'Correlation between {ts_column1} and {ts_column2}: {correlation:.2f}')
            plt.show()
        else:
            plt.close()

        return correlation

    # TODO: fit underling distribution and plot in one plot both the distribution and the histogram of the data
    
    # TODO: Find the best p,d,q for ARIMA model and plot the k next steps with the AIC and BIC 
    
    # TODO : Find the best (p,d,q), (P,D,[s],Q) for SARIMA model and plot the k next steps with the AIC and BIC

    # Save the plot
    def save_plot(self,file_name ,save_path: str):
        '''
        Save the plot to the specified path.
        '''
        if not save_path:
            save_path = os.path.join('Output', 'Plots','')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path+f'{file_name}_plot.png')