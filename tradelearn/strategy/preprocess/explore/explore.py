import pandas as pd
import math
import base64
from io import BytesIO
import seaborn as sns
import os
cur_dir_path = os.path.abspath(os.path.dirname(__file__))

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt

from typing import Any, Literal, Dict

import json
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import warnings

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


class Explore:

    def __init__(self, df:pd.DataFrame, date_column:str) -> None:
        if (date_column not in df.columns)&(date_column not in df.index.names):
            raise ValueError(f"{date_column} not found in DataFrame columns")
        
        if date_column not in df.index.names:
            df.set_index(date_column, inplace=True)
        
        self.data = df.select_dtypes(include=['number'])

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

        self.line_colors = theme.get('colors', {}).get('line_colors', [])


    def _nullity_sort(self, df, sort=None, axis='columns'):
        """
        Sorts a DataFrame according to its nullity, in either ascending or descending order.

        :param df: The DataFrame object being sorted.
        :param sort: The sorting method: either "ascending", "descending", or None (default).
        :return: The nullity-sorted DataFrame.
        """
        if sort is None:
            return df
        elif sort not in ['ascending', 'descending']:
            raise ValueError('The "sort" parameter must be set to "ascending" or "descending".')

        if axis not in ['rows', 'columns']:
            raise ValueError('The "axis" parameter must be set to "rows" or "columns".')

        if axis == 'columns':
            if sort == 'ascending':
                return df.iloc[np.argsort(df.count(axis='columns').values), :]
            elif sort == 'descending':
                return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
        elif axis == 'rows':
            if sort == 'ascending':
                return df.iloc[:, np.argsort(df.count(axis='rows').values)]
            elif sort == 'descending':
                return df.iloc[:, np.flipud(np.argsort(df.count(axis='rows').values))]

    def _nullity_filter(self, df, filter=None, p=0, n=0):
        """
        Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
        percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
        to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
        `nullity_filter(df, filter='top', p=.75, n=5)`.

        :param df: The DataFrame whose columns are being filtered.
        :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
        or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
        as None.
        :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
        completeness. Input should be in the range [0, 1].
        :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
        :return: The nullity-filtered `DataFrame`.
        """
        if filter == 'top':
            if p:
                df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
            if n:
                df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
        elif filter == 'bottom':
            if p:
                df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
            if n:
                df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
        return df
    
    def matrix(
        self, filter=None, n=0, p=0, sort=None, figsize=(25, 10), width_ratios=(15, 1),
        color=(0.25, 0.25, 0.25), fontsize=16, labels=None, label_rotation=45, sparkline=True,
        freq=None, ax=None
    ):
        """
        A matrix visualization of the nullity of the given DataFrame.

        :param df: The `DataFrame` being mapped.
        :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
        :param n: The max number of columns to include in the filtered DataFrame.
        :param p: The max percentage fill of the columns in the filtered DataFrame.
        :param sort: The row sort order to apply. Can be "ascending", "descending", or None.
        :param figsize: The size of the figure to display.
        :param fontsize: The figure's font size. Default to 16.
        :param labels: Whether or not to display the column names. Defaults to the underlying data labels when there are
            50 columns or less, and no labels when there are more than 50 columns.
        :param label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
        :param sparkline: Whether or not to display the sparkline. Defaults to True.
        :param width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`.
            Does nothing if `sparkline=False`.
        :param color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
        :return: The plot axis.
        """
        df = self.data
        df = self._nullity_filter(df, filter=filter, n=n, p=p)
        df = self._nullity_sort(df, sort=sort, axis='columns')

        height = df.shape[0]
        width = df.shape[1]

        # z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
        z = df.notnull().values
        g = np.zeros((height, width, 3), dtype=np.float32)

        g[z < 0.5] = [1, 1, 1]
        g[z > 0.5] = color

        # Set up the matplotlib grid layout. A unary subplot if no sparkline, a left-right splot if yes sparkline.
        if ax is None:
            plt.figure(figsize=figsize)
            if sparkline:
                gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
                gs.update(wspace=0.08)
                ax1 = plt.subplot(gs[1])
            else:
                gs = gridspec.GridSpec(1, 1)
            ax0 = plt.subplot(gs[0])
        else:
            if sparkline is not False:
                warnings.warn(
                    "Plotting a sparkline on an existing axis is not currently supported. "
                    "To remove this warning, set sparkline=False."
                )
                sparkline = False
            ax0 = ax

        # Create the nullity plot.
        ax0.imshow(g, interpolation='none')

        # Remove extraneous default visual elements.
        ax0.set_aspect('auto')
        ax0.grid(visible=False)
        ax0.xaxis.tick_top()
        ax0.xaxis.set_ticks_position('none')
        ax0.yaxis.set_ticks_position('none')
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['left'].set_visible(False)

        # Set up and rotate the column ticks. The labels argument is set to None by default. If the user specifies it in
        # the argument, respect that specification. Otherwise display for <= 50 columns and do not display for > 50.
        if labels or (labels is None and len(df.columns) <= 50):
            ha = 'left'
            ax0.set_xticks(list(range(0, width)))
            ax0.set_xticklabels(list(df.columns), rotation=label_rotation, ha=ha, fontsize=fontsize)
        else:
            ax0.set_xticks([])

        # Adds Timestamps ticks if freq is not None, else set up the two top-bottom row ticks.
        if freq:
            ts_list = []

            if type(df.index) == pd.PeriodIndex:
                ts_array = pd.date_range(df.index.to_timestamp().date[0],
                                         df.index.to_timestamp().date[-1],
                                         freq=freq).values

                ts_ticks = pd.date_range(df.index.to_timestamp().date[0],
                                         df.index.to_timestamp().date[-1],
                                         freq=freq).map(lambda t:
                                                        t.strftime('%Y-%m-%d'))

            elif type(df.index) == pd.DatetimeIndex:
                ts_array = pd.date_range(df.index[0], df.index[-1],
                                         freq=freq).values

                ts_ticks = pd.date_range(df.index[0], df.index[-1],
                                         freq=freq).map(lambda t:
                                                        t.strftime('%Y-%m-%d'))
            else:
                raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
            try:
                for value in ts_array:
                    ts_list.append(df.index.get_loc(value))
            except KeyError:
                raise KeyError('Could not divide time index into desired frequency.')

            ax0.set_yticks(ts_list)
            ax0.set_yticklabels(ts_ticks, fontsize=int(fontsize / 16 * 20), rotation=0)
        else:
            ax0.set_yticks([0, df.shape[0] - 1])
            ax0.set_yticklabels([1, df.shape[0]], fontsize=int(fontsize / 16 * 20), rotation=0)

        # Create the inter-column vertical grid.
        in_between_point = [x + 0.5 for x in range(0, width - 1)]
        for in_between_point in in_between_point:
            ax0.axvline(in_between_point, linestyle='-', color='white')

        if sparkline:
            # Calculate row-wise completeness for the sparkline.
            completeness_srs = df.notnull().astype(bool).sum(axis=1)
            x_domain = list(range(0, height))
            y_range = list(reversed(completeness_srs.values))
            min_completeness = min(y_range)
            max_completeness = max(y_range)
            min_completeness_index = y_range.index(min_completeness)
            max_completeness_index = y_range.index(max_completeness)

            # Set up the sparkline, remove the border element.
            ax1.grid(visible=False)
            ax1.set_aspect('auto')
            # GH 25
            if int(mpl.__version__[0]) <= 1:
                ax1.set_axis_bgcolor((1, 1, 1))
            else:
                ax1.set_facecolor((1, 1, 1))
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.set_ymargin(0)

            # Plot sparkline---plot is sideways so the x and y axis are reversed.
            ax1.plot(y_range, x_domain, color=color)

            if labels:
                # Figure out what case to display the label in: mixed, upper, lower.
                label = 'Data Completeness'
                if str(df.columns[0]).islower():
                    label = label.lower()
                if str(df.columns[0]).isupper():
                    label = label.upper()

                # Set up and rotate the sparkline label.
                ha = 'left'
                ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
                ax1.set_xticklabels([label], rotation=label_rotation, ha=ha, fontsize=fontsize)
                ax1.xaxis.tick_top()
                ax1.set_yticks([])
            else:
                ax1.set_xticks([])
                ax1.set_yticks([])

            # Add maximum and minimum labels, circles.
            ax1.annotate(max_completeness,
                         xy=(max_completeness, max_completeness_index),
                         xytext=(max_completeness + 2, max_completeness_index),
                         fontsize=int(fontsize / 16 * 14),
                         va='center',
                         ha='left')
            ax1.annotate(min_completeness,
                         xy=(min_completeness, min_completeness_index),
                         xytext=(min_completeness - 2, min_completeness_index),
                         fontsize=int(fontsize / 16 * 14),
                         va='center',
                         ha='right')

            ax1.set_xlim([min_completeness - 2, max_completeness + 2])  # Otherwise the circles are cut off.
            ax1.plot([min_completeness], [min_completeness_index], '.', color=color, markersize=10.0)
            ax1.plot([max_completeness], [max_completeness_index], '.', color=color, markersize=10.0)

            # Remove tick mark (only works after plotting).
            ax1.xaxis.set_ticks_position('none')

        plt.title("Missing Value Distribute")
        return ax0
    
    def _corr_selector(
        self,
        corr: pd.Series | pd.DataFrame,
        split: Literal["pos", "neg", "high", "low"] | None = None,
        threshold: float = 0,
    ) -> pd.Series | pd.DataFrame:
        """Select the desired correlations using this utility function.

        Parameters
        ----------
        corr : pd.Series | pd.DataFrame
            pd.Series or pd.DataFrame of correlations
        split : Optional[str], optional
            Type of split performed, by default None
                * {None, "pos", "neg", "high", "low"}
        threshold : float, optional
            Value between 0 and 1 to set the correlation threshold, by default 0 unless \
            split = "high" or split = "low", in which case default is 0.3

        Returns
        -------
        pd.DataFrame
            List or matrix of (filtered) correlations
        """
        if split == "pos":
            corr = corr.where((corr >= threshold) & (corr > 0))
            print(
                'Displaying positive correlations. Specify a positive "threshold" to '
                "limit the results further.",
            )
        elif split == "neg":
            corr = corr.where((corr <= threshold) & (corr < 0))
            print(
                'Displaying negative correlations. Specify a negative "threshold" to '
                "limit the results further.",
            )
        elif split == "high":
            threshold = 0.3 if threshold <= 0 else threshold
            corr = corr.where(np.abs(corr) >= threshold)
            print(
                f"Displaying absolute correlations above the threshold ({threshold}). "
                'Specify a positive "threshold" to limit the results further.',
            )
        elif split == "low":
            threshold = 0.3 if threshold <= 0 else threshold
            corr = corr.where(np.abs(corr) <= threshold)
            print(
                f"Displaying absolute correlations below the threshold ({threshold}). "
                'Specify a positive "threshold" to limit the results further.',
            )

        return corr
    
    def corr_mat(
        self,
        data: pd.DataFrame,
        split: Literal["pos", "neg", "high", "low"] | None = None,
        threshold: float = 0,
        target: pd.DataFrame | pd.Series | np.ndarray | str | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        colored: bool = True,
    ) -> pd.DataFrame | pd.Series:
        """Return a color-encoded correlation matrix.

        Parameters
        ----------
        data : pd.DataFrame
            2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
            is provided, the index/column information is used to label the plots
        split : Optional[Literal['pos', 'neg', 'high', 'low']], optional
            Type of split to be performed, by default None
            {None, "pos", "neg", "high", "low"}
        threshold : float, optional
            Value between 0 and 1 to set the correlation threshold, by default 0 unless \
            split = "high" or split = "low", in which case default is 0.3
        target : Optional[pd.DataFrame | str], optional
            Specify target for correlation. E.g. label column to generate only the \
            correlations between each feature and the label, by default None
        method : Literal['pearson', 'spearman', 'kendall'], optional
            method: {"pearson", "spearman", "kendall"}, by default "pearson"
            * pearson: measures linear relationships and requires normally distributed \
                and homoscedastic data.
            * spearman: ranked/ordinal correlation, measures monotonic relationships.
            * kendall: ranked/ordinal correlation, measures monotonic relationships. \
                Computationally more expensive but more robust in smaller dataets than \
                "spearman"
        colored : bool, optional
            If True the negative values in the correlation matrix are colored in red, by \
            default True

        Returns
        -------
        pd.DataFrame | pd.Styler
            If colored = True - corr: Pandas Styler object
            If colored = False - corr: Pandas DataFrame

        """
        def color_negative_red(val: float) -> str:
            color = "#FF3344" if val < 0 else None
            return f"color: {color}"

        data = pd.DataFrame(data)

        if isinstance(target, (str, list, pd.Series, np.ndarray)):
            target_data = []
            if isinstance(target, str):
                target_data = data[target]
                data = data.drop(target, axis=1)

            elif isinstance(target, (list, pd.Series, np.ndarray)):
                target_data = pd.Series(target)
                target = target_data.name

            corr = pd.DataFrame(
                data.corrwith(target_data, method=method, numeric_only=True),
            )
            corr = corr.sort_values(corr.columns[0], ascending=False)
            corr.columns = [target]

        else:
            corr = data.corr(method=method, numeric_only=True)

        corr = self._corr_selector(corr, split=split, threshold=threshold)

        if colored:
            return corr.style.applymap(color_negative_red).format("{:.2f}", na_rep="-")
        return corr
    
    def corr_plot(
        self,
        split: Literal["pos", "neg", "high", "low"] | None = None,
        threshold: float = 0,
        target: pd.Series | str | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        cmap: str = "BrBG",
        figsize: tuple[float, float] = (12, 10),
        annot: bool = True,
        dev: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> plt.Axes:
        """2D visualization of the correlation between feature-columns excluding NA values.

        Parameters
        ----------
        data : pd.DataFrame
            2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
            is provided, the index/column information is used to label the plots
        split : Optional[str], optional
            Type of split to be performed {None, "pos", "neg", "high", "low"}, by default \
            None
                * None: visualize all correlations between the feature-columns
                * pos: visualize all positive correlations between the feature-columns \
                    above the threshold
                * neg: visualize all negative correlations between the feature-columns \
                    below the threshold
                * high: visualize all correlations between the feature-columns for \
                    which abs (corr) > threshold is True
                * low: visualize all correlations between the feature-columns for which \
                    abs(corr) < threshold is True

        threshold : float, optional
            Value between 0 and 1 to set the correlation threshold, by default 0 unless \
                split = "high" or split = "low", in which case default is 0.3
        target : Optional[pd.Series | str], optional
            Specify target for correlation. E.g. label column to generate only the \
            correlations between each feature and the label, by default None
        method : Literal['pearson', 'spearman', 'kendall'], optional
            method: {"pearson", "spearman", "kendall"}, by default "pearson"
                * pearson: measures linear relationships and requires normally \
                    distributed and homoscedastic data.
                * spearman: ranked/ordinal correlation, measures monotonic relationships.
                * kendall: ranked/ordinal correlation, measures monotonic relationships. \
                    Computationally more expensive but more robust in smaller dataets \
                    than "spearman".

        cmap : str, optional
            The mapping from data values to color space, matplotlib colormap name or \
            object, or list of colors, by default "BrBG"
        figsize : tuple[float, float], optional
            Use to control the figure size, by default (12, 10)
        annot : bool, optional
            Use to show or hide annotations, by default True
        dev : bool, optional
            Display figure settings in the plot by setting dev = True. If False, the \
            settings are not displayed, by default False

        kwargs : optional
            Additional elements to control the visualization of the plot, e.g.:

                * mask: bool, default True
                    If set to False the entire correlation matrix, including the upper \
                    triangle is shown. Set dev = False in this case to avoid overlap.
                * vmax: float, default is calculated from the given correlation \
                    coefficients.
                    Value between -1 or vmin <= vmax <= 1, limits the range of the cbar.
                * vmin: float, default is calculated from the given correlation \
                    coefficients.
                    Value between -1 <= vmin <= 1 or vmax, limits the range of the cbar.
                * linewidths: float, default 0.5
                    Controls the line-width inbetween the squares.
                * annot_kws: dict, default {"size" : 10}
                    Controls the font size of the annotations. Only available when \
                    annot = True.
                * cbar_kws: dict, default {"shrink": .95, "aspect": 30}
                    Controls the size of the colorbar.
                * Many more kwargs are available, i.e. "alpha" to control blending, or \
                    options to adjust labels, ticks ...

            Kwargs can be supplied through a dictionary of key-value pairs (see above).

        Returns
        -------
        ax: matplotlib Axes
            Returns the Axes object with the plot for further tweaking.

        """

        data = self.data

        corr = self.corr_mat(
            data,
            split=split,
            threshold=threshold,
            target=target,
            method=method,
            colored=False,
        )

        mask = np.zeros_like(corr, dtype=bool)

        if target is None:
            mask = np.triu(np.ones_like(corr, dtype=bool))

        vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
        vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)

        fig, ax = plt.subplots(figsize=figsize)

        # Specify kwargs for the heatmap
        kwargs = {
            "mask": mask,
            "cmap": cmap,
            "annot": annot,
            "vmax": vmax,
            "vmin": vmin,
            "linewidths": 0.5,
            "annot_kws": {"size": 10},
            "cbar_kws": {"shrink": 0.95, "aspect": 30},
            **kwargs,
        }

        # Draw heatmap with mask and default settings
        sns.heatmap(corr, center=0, fmt=".2f", **kwargs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', va='top', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=-45, ha='right', va='bottom', fontsize=10)
        ax.set_title(f"Variable Correlation ({method})", fontsize=15)

        # Settings
        if dev:
            fig.suptitle(
                f"\
                Settings (dev-mode): \n\
                - split-mode: {split} \n\
                - threshold: {threshold} \n\
                - method: {method} \n\
                - annotations: {annot} \n\
                - cbar: \n\
                    - vmax: {vmax} \n\
                    - vmin: {vmin} \n\
                - linewidths: {kwargs['linewidths']} \n\
                - annot_kws: {kwargs['annot_kws']} \n\
                - cbar_kws: {kwargs['cbar_kws']}",
                fontsize=12,
                color="gray",
                x=0.35,
                y=0.85,
                ha="left",
            )

        return ax

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
        plt.ylabel(f'{ts_column}')
        plt.title(f'{ts_column}: Over Time')
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
    def histogram_plot(self,ts_column:str,display=True, save:bool=False, save_path:str=None,bin_size:int=100 , ax=None):
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
        if ax:
            ax.grid(False)
            ax.hist(self.data[ts_column], bins=bin_size)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{ts_column}: Histogram')
        else:
            plt.grid(False)
            plt.hist(self.data[ts_column],bins=bin_size)
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
        plt.xticks([]) 
        plt.ylabel(f'{ts_column}')
        plt.title(f'{ts_column}: Box Plot')
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
    
    def report(self, filename: str = './explore.html'):
        def template(template_name: str):
            env = Environment(loader=FileSystemLoader(os.path.join(cur_dir_path, 'template')))
            return env.get_template(template_name)
        
        def fig_to_base64_img(fig):
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            return img_base64

        # PART0
        non_missing_count = self.data.notnull().sum().tolist()
        missing_count = self.data.isnull().sum().tolist()

        columns_info = []
        for i in range(self.data.shape[1]):
            col_info = {
                "name": self.data.columns[i],
                "dtype": str(self.data.dtypes[i]),
                "non_missing": non_missing_count[i],
                "missing": missing_count[i],
                "missing_ratio": missing_count[i]/self.data.shape[0]
            }
            columns_info.append(col_info)

        fig = plt.figure(figsize=(8, 6))
        self.matrix()
        plt.tight_layout()
        mi_plot_html = fig_to_base64_img(plt)
        plt.close(fig)

        overview_html = template('content/overview.html').render(total_rows=self.data.shape[0],
                                                                 total_columns=self.data.shape[1],
                                                                 columns_info=columns_info,
                                                                 img={"img_base64": mi_plot_html, "alt": "Miss Information"})

        # PART1
        data = self.data.copy()
        describe_html = data.describe().T.round(2).to_html()

        # PART2
        ncols = 4
        nrows = math.ceil(len(data.columns) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows), constrained_layout=True)
        axes = axes.flatten()

        for i, col in enumerate(data.columns):
            self.histogram_plot(col, False, ax=axes[i])
            axes[i].set_title(f'{col}', fontsize=15)
            axes[i].set_ylabel("Frequency", fontsize=15)
            axes[i].tick_params(axis='both', labelsize=10)
            
            if i % ncols != 0:
                axes[i].set_ylabel("")

        for i in range(len(data.columns), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle("Variable Histograms", fontsize=20, fontweight='bold', y=1.04)
                     
        hist_plot_html = fig_to_base64_img(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        self.corr_plot()
        plt.tight_layout()
        corr_plot_html = fig_to_base64_img(plt)
        plt.close(fig)
        
        img_list = [
            {"img_base64": hist_plot_html, "alt": "Histogram"},
            {"img_base64": corr_plot_html, "alt": "Correlation Heatmap"},
        ]
        variable_html = template('content/variable.html').render(img_list=img_list)

        # # PART3
        render_data_list = []

        for col in data.columns:
            # 折线图
            fig = plt.figure(figsize=(6, 4))
            self.histogram_plot(col,False)
            plt.ylabel('some numbers')
            curve_plot = fig_to_base64_img(fig)
            plt.close(fig)

            # 箱型图
            fig = plt.figure(figsize=(6, 4))
            self.box_plot(col, False)
            box_plot = fig_to_base64_img(fig)
            plt.close(fig)

            # 箱型图
            fig = plt.figure(figsize=(6, 4))
            self.line_plot(col, False)
            line_plot = fig_to_base64_img(fig)
            plt.close(fig)

            # 描述性统计
            stats = {
                'mean': f"{data[col].mean():.4f}",
                'std': f"{data[col].std():.4f}",
                'skew': f"{data[col].skew():.4f}",
                'kurt': f"{data[col].kurt():.4f}",
                'min': f"{data[col].min():.4f}",
                'quantile25': f"{data[col].quantile(0.25):.4f}",  # Q1
                'quantile50': f"{data[col].median():.4f}",        # Q2 (中位数)
                'quantile75': f"{data[col].quantile(0.75):.4f}",  # Q3
                'max': f"{data[col].max():.4f}",
            }

            # 时间序列图（仅限 'open'）
            ts_plot = ""
            if col == 'close':
                fig = self.full_statistics_plot('open', display=False, save=False)
                ts_plot = fig_to_base64_img(fig)
                plt.close(fig)

            # 收集到列表中
            render_data_list.append({
                "col_name": col,
                "stats": stats,
                "curve_plot": curve_plot.split(",")[-1],
                "box_plot": box_plot.split(",")[-1],
                "line_plot": line_plot.split(",")[-1],
                "ts_plot": ts_plot.split(",")[-1] if ts_plot else ""
            })

        detail_html = template('content/detail.html').render(variables=render_data_list)
    
        # PRIME
        part0_html = template('section.html').render(section_content=overview_html, section_title='Data Overview',
                                                section_anchor_id='part0')
        part1_html = template('section.html').render(section_content=describe_html, section_title='Descriptive Statistics',
                                                section_anchor_id='part1')
        part2_html = template('section.html').render(section_content=variable_html, section_title='Variable Overview',
                                                section_anchor_id='part2')
        part3_html = template('section.html').render(section_content=detail_html, section_title='Variable Details',
                                                section_anchor_id='part3')
        html = template('main.html').render(title='Exploratory Analysis Report',
                                               part0_html=part0_html,
                                               part1_html=part1_html,
                                               part2_html=part2_html,
                                               part3_html=part3_html,
                                               p1=len(part1_html) > 0,
                                               p2=len(part2_html) > 0,
                                               p3=len(part3_html) > 0,
                                               variables=[var['col_name'] for var in render_data_list])
        
        with open(filename, 'w+', encoding='utf8') as file:
            file.write(html)


