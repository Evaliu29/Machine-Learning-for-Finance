
import numpy as np
import pandas as pd
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.data import morningstar as mstar


# Custom Factor 1 : Dividend Yield
class Div_Yield(CustomFactor):
    inputs = [morningstar.valuation_ratios.dividend_yield]
    window_length = 1

    def compute(self, today, assets, out, d_y):
        out[:] = d_y[-1]


# Custom Factor 2 : P/B Ratio
class Price_to_Book(CustomFactor):
    inputs = [morningstar.valuation_ratios.pb_ratio]
    window_length = 1

    def compute(self, today, assets, out, p_b_r):
        out[:] = p_b_r[-1]


# Custom Factor 3 : Price to Trailing 12 Month Sales
class Price_to_TTM_Sales(CustomFactor):
    inputs = [morningstar.valuation_ratios.ps_ratio]
    window_length = 1

    def compute(self, today, assets, out, ps):
        out[:] = -ps[-1]


# Custom Factor 4 : Price to Trailing 12 Month Cashflow
class Price_to_TTM_Cashflows(CustomFactor):
    inputs = [morningstar.valuation_ratios.pcf_ratio]
    window_length = 1

    def compute(self, today, assets, out, pcf):
        out[:] = -pcf[-1]

    # Factor 5: PE ratio


class PE(CustomFactor):
    inputs = [morningstar.valuation_ratios.pe_ratio]
    window_length = 1

    def compute(self, today, assets, out, p_e):
        out[:] = p_e[-1]


# PEG factor
class PEG(CustomFactor):
    inputs = [morningstar.valuation_ratios.peg_ratio]
    window_length = 1

    def compute(self, today, assets, out, peg):
        out[:] = peg[-1]
        out[:] = np.nan_to_num(out[:])


##Quality
class Quality(CustomFactor):
    inputs = [morningstar.operation_ratios.roe]
    window_length = 1

    def compute(self, today, assets, out, roe):
        out[:] = roe[-1] 


# Volatility
class Volatility(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
        close = pd.DataFrame(data=close, columns=assets)
        # Since we are going to rank largest is best we need to invert the sdev.
        out[:] = 1 / np.log(close).diff().std()


# This factor creates the synthetic S&P500
class SPY_proxy(CustomFactor):
    inputs = [morningstar.valuation.market_cap]
    window_length = 1

    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]


# roe > 0.08
# This pulls all necessary data in one step
def Data_Pull():
    # Quality() > 0.08

    pipe_columns = {
        'DY': Div_Yield(),
        'Price/TTM sales': Price_to_TTM_Sales(),
        'PEG': PEG(),
        'PE': PE(),
        'Price to Book': Price_to_Book(),
        'Quality': Quality(),
        'SPY Proxy': SPY_proxy(),
        'Price / TTM Cashflow': Price_to_TTM_Cashflows(),
        'Volatility': Volatility()
    }

    # create the pipeline for the data pull
    Data_Pipe = Pipeline(columns=pipe_columns)
    return Data_Pipe


# function to filter out unwanted values in the scores
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x


def standard_frame_compute(df):
    """
    Standardizes the Pipeline API data pull
    using the S&P500's means and standard deviations for
    particular CustomFactors.

    parameters
    ----------
    df: numpy.array
        full result of Data_Pull

    returns
    -------
    numpy.array
        standardized Data_Pull results

    numpy.array
        index of equities
    """

    # basic clean of dataset to remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)

    # create separate dataframe for SPY
    # to store standardization values
    df_SPY = df_SPY.head(500)

    # get dataframes into numpy array
    df_SPY = df_SPY.as_matrix()

    # store index values
    index = df.index.values

    # turn iinto a numpy array for speed
    df = df.as_matrix()

    # create an empty vector on which to add standardized values
    df_standard = np.empty(df.shape[0])

    for col_SPY, col_full in zip(df_SPY.T, df.T):
        # summary stats for S&P500
        mu = np.mean(col_SPY)
        sigma = np.std(col_SPY)
        col_standard = np.array(((col_full - mu) / sigma))

        # create vectorized function (lambda equivalent)
        fltr = np.vectorize(filter_fn)
        col_standard = (fltr(col_standard))

        # make range between -10 and 10
        col_standard = (col_standard / df.shape[1])

        # attach calculated values as new row in df_standard
        df_standard = np.vstack((df_standard, col_standard))

    # get rid of first entry (empty scores)
    df_standard = np.delete(df_standard, 0, 0)

    return (df_standard, index)


def composite_score(df, index):
    """
    Summarize standardized data in a single number.

    parameters
    ----------
    df: numpy.array
        standardized results

    index: numpy.array
        index of equities

    returns
    -------
    pandas.Series
        series of summarized, ranked results

    """

    # sum up transformed data
    df_composite = df.sum(axis=0)

    # put into a pandas dataframe and connect numbers
    # to equities via reindexing
    df_composite = pd.Series(data=df_composite, index=index)

    # sort descending
    df_composite.sort(ascending=False)

    return df_composite


def initialize(context):
    # get data from pipeline
    data_pull = Data_Pull()
    attach_pipeline(data_pull, 'Data')

    # filter out bad stocks for universe
    roe = morningstar.operation_ratios.roe.latest
    f = roe > 0.1
    # netprofitmargin figure
    # netmargin = morningstar.operation_ratios.net_margin.latest
    # f2 = netmargin > 0.08
    mask = filter_universe()
    data_pull.set_screen(mask & f)

    # set leverage ratios for longs and shorts
    context.long_leverage = 1.3
    context.short_leverage = -0.3

    # at the start of each moth, run the rebalancing function
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open(minutes=30))

    # clean untradeable securities daily
    schedule_function(daily_clean,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))
    pass


# called before every day of trading
def before_trading_start(context, data):
    # apply the logic to the data pull in order to get a ranked list of equities
    context.output = pipeline_output('Data')
    context.output, index = standard_frame_compute(context.output)
    context.output = composite_score(context.output, index)

    # create lists of stocks on which to go long and short
    context.long_set = set(context.output.head(26).index)
    context.short_set = set(context.output.tail(6).index)


# log long and short equities and their corresponding composite scores
def handle_data(context, data):
    """
    print "LONG LIST"
    log.info(context.long_set)

    print "SHORT LIST"
    log.info(context.short_set)
    """
    pass


# called at the start of every month in order to rebalance the longs and shorts lists
def rebalance(context, data):
    # calculate how much of each stock to buy or hold
    long_pct = context.long_leverage / len(context.long_set)
    short_pct = context.short_leverage / len(context.short_set)

    # universe now contains just longs and shorts
    context.security_set = set(context.long_set.union(context.short_set))

    for stock in context.security_set:
        if data.can_trade(stock):
            if stock in context.long_set:
                order_target_percent(stock, long_pct)
            elif stock in context.short_set:
                order_target_percent(stock, short_pct)

    # close out stale positions
    daily_clean(context, data)


# make sure all untradeable securities are sold off each day
def daily_clean(context, data):
    for stock in context.portfolio.positions:
        if stock not in context.security_set and data.can_trade(stock):
            order_target_percent(stock, 0)


def filter_universe():
    """
    9 filters:
        1. common stock
        2 & 3. not limited partnership - name and database check
        4. database has fundamental data
        5. not over the counter
        6. not when issued
        7. not depository receipts
        8. primary share
        9. high dollar volume
    Check Scott's notebook for more details.
    """
    common_stock = mstar.share_class_reference.security_type.latest.eq('ST00000001')
    not_lp_name = ~mstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    not_lp_balance_sheet = mstar.balance_sheet.limited_partnership.latest.isnull()
    have_data = mstar.valuation.market_cap.latest.notnull()
    not_otc = ~mstar.share_class_reference.exchange_id.latest.startswith('OTC')
    not_wi = ~mstar.share_class_reference.symbol.latest.endswith('.WI')
    not_depository = ~mstar.share_class_reference.is_depositary_receipt.latest
    primary_share = IsPrimaryShare()

    # Combine the above filters.
    tradable_filter = (common_stock & not_lp_name & not_lp_balance_sheet &
                       have_data & not_otc & not_wi & not_depository & primary_share)

    high_volume_tradable = (AverageDollarVolume(window_length=21,
                                                mask=tradable_filter).percentile_between(70, 100))

    screen = high_volume_tradable

    return screen

##factor analysis one by one 
#original factor
import alphalens as al
import numpy as np
import pandas as pd
from quantopian.research import run_pipeline
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, Returns
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.filters import QTradableStocksUS
from time import time
from quantopian.pipeline.data import morningstar

# Alphalens takes your factor and examines how useful it is for predicting relative value through a collection of different metrics. It breaks all the stocks in your chosen universe into different quantiles based on their ranking according to your factor and analyzes the returns, information coefficient (IC), the turnover of each quantile, and provides a breakdown of returns and IC by sector.
#
# Throughout the course of this lecture we will detail how to interpret the various individual plots generated by an `Alphalens` tear sheet and include the proper call to generate the whole tear sheet at once at the end.

# ## Sector Codes
#
# These are the possible sector codes for each security, as given by Morningstar. We will use this dictionary to help categorize our results as we walk through a factor analysis so that we can break out our information by sector.

# In[3]:


MORNINGSTAR_SECTOR_CODES = {
    -1: 'Misc',
    101: 'Basic Materials',
    102: 'Consumer Cyclical',
    103: 'Financial Services',
    104: 'Real Estate',
    205: 'Consumer Defensive',
    206: 'Healthcare',
    207: 'Utilities',
    308: 'Communication Services',
    309: 'Energy',
    310: 'Industrials',
    311: 'Technology',
}

# ## Defining a universe
#
# As always, we need to define our universe. In this case we use the QTradableStocksUS, as seen in the forums [here](https://www.quantopian.com/posts/working-on-our-best-universe-yet-qtradablestocksus).

# In[4]:


universe = QTradableStocksUS()


# In[5]:


def make_factors():
    # Custom Factor 1 : Dividend Yield
    class Div_Yield(CustomFactor):
        inputs = [morningstar.valuation_ratios.dividend_yield]
        window_length = 1

        def compute(self, today, assets, out, d_y):
            out[:] = d_y[-1]

    # Custom Factor 2 : P/B Ratio
    class Price_to_Book(CustomFactor):
        inputs = [morningstar.valuation_ratios.pb_ratio]
        window_length = 1

        def compute(self, today, assets, out, p_b_r):
            out[:] = -p_b_r[-1]

    # Custom Factor 3 : Price to Trailing 12 Month Sales
    class Price_to_TTM_Sales(CustomFactor):
        inputs = [morningstar.valuation_ratios.ps_ratio]
        window_length = 1

        def compute(self, today, assets, out, ps):
            out[:] = -ps[-1]

    # Custom Factor 4 : Price to Trailing 12 Month Cashflow
    class Price_to_TTM_Cashflows(CustomFactor):
        inputs = [morningstar.valuation_ratios.pcf_ratio]
        window_length = 1

        def compute(self, today, assets, out, pcf):
            out[:] = -pcf[-1]

    class SPY_proxy(CustomFactor):
        inputs = [morningstar.valuation.market_cap]
        window_length = 1

        def compute(self, today, assets, out, mc):
            out[:] = mc[-1]

    return {
        'Price / TTM Sales': Price_to_TTM_Sales,
        'Dividen Yield': Div_Yield,
        'Price to Book': Price_to_Book,
        'SPY Proxy': SPY_proxy,
        'Price / TTM Cashflow': Price_to_TTM_Cashflows,
    }


# pipe_columns = {
#         'Price / TTM Sales':Price_to_TTM_Sales(),
#         'Dividen Yield':Div_Yield(),
#         'Price to Book':Price_to_Book(),
#         'SPY Proxy':SPY_proxy(),
#         'Price / TTM Cashflow':Price_to_TTM_Cashflows()
#     }

#     # create the pipeline for the data pull
#     Data_Pipe = Pipeline(columns = pipe_columns)


# In[6]:


# from scipy import stats

factors = make_factors()

combined_alpha = None

for name, f in factors.items():
    pipe = Pipeline(
    columns={
        'CombinedAlpha': f(mask=universe),
        'Sector': Sector()
    },
    screen=universe)
    
    start_timer = time()
    results = run_pipeline(pipe, '2014-01-01', '2016-08-29')
    end_timer = time()
    results = results.fillna(value=0);

    # In[8]:


    # results['CombinedAlpha'].describe()


    # In[9]:


    my_factor = results['CombinedAlpha']
    sectors = results['Sector']
    asset_list = results.index.levels[1].unique()
    prices = get_pricing(asset_list, start_date='2014-01-01', end_date='2018-08-29', fields='open_price')
    periods = (1, 5, 10)

    # In[10]:


    # my_factor


    # In[11]:


    factor_data = al.utils.get_clean_factor_and_forward_returns(factor=my_factor,
                                                                prices=prices,
                                                                groupby=sectors,
                                                                quantiles=2,
                                                                groupby_labels=MORNINGSTAR_SECTOR_CODES,
                                                                periods=periods)

    # In[12]:


    al.tears.create_full_tear_sheet(factor_data, by_group=True);

    
# In[7]:


 




# ## my model analysis
def make_factors():

    class PE(CustomFactor):
        inputs = [morningstar.valuation_ratios.pe_ratio]
        window_length = 1

        def compute(self, today, assets, out, p_e):
            out[:] = p_e[-1]

    # PEG factor
    class PEG(CustomFactor):
        inputs = [morningstar.valuation_ratios.peg_ratio]
        window_length = 1

        def compute(self, today, assets, out, peg):
            out[:] = peg[-1]
            out[:] = np.nan_to_num(out[:])

    ##Quality
    class Quality(CustomFactor):
        inputs = [morningstar.operation_ratios.roe]
        window_length = 1

        def compute(self, today, assets, out, roe):
            out[:] = roe[-1] > 0.08

    # Volatility
    class Volatility(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):
            close = pd.DataFrame(data=close, columns=assets)
            # Since we are going to rank largest is best we need to invert the sdev.
            out[:] = 1 / np.log(close).diff().std()

    # This factor creates the synthetic S&P500
    class SPY_proxy(CustomFactor):
        inputs = [morningstar.valuation.market_cap]
        window_length = 1

        def compute(self, today, assets, out, mc):
            out[:] = mc[-1]

    return {
#         'DY': Div_Yield,
#         'Price/TTM sales': Price_to_TTM_Sales,
        'PEG': PEG,
        'PE': PE,
#         'Price to Book': Price_to_Book,
        'Quality': Quality,
#         'SPY Proxy': SPY_proxy,
#         'Price / TTM Cashflow': Price_to_TTM_Cashflows,
        'Volatility': Volatility
    }


# pipe_columns = {
#         'Price / TTM Sales':Price_to_TTM_Sales(),
#         'Dividen Yield':Div_Yield(),
#         'Price to Book':Price_to_Book(),
#         'SPY Proxy':SPY_proxy(),
#         'Price / TTM Cashflow':Price_to_TTM_Cashflows()
#     }

#     # create the pipeline for the data pull
#     Data_Pipe = Pipeline(columns = pipe_columns)


# In[14]:


# from scipy import stats
roe = morningstar.operation_ratios.roe.latest
fil = roe > 0.1
mask = universe
factors = make_factors()

combined_alpha = None

factors = make_factors()

combined_alpha = None

for name, f in factors.items():
    pipe = Pipeline(
    columns={
        'CombinedAlpha': f(mask=universe),
        'Sector': Sector()
    },
    screen=universe & fil)
    
    start_timer = time()
    results = run_pipeline(pipe, '2014-01-01', '2016-08-29')
    end_timer = time()
    results = results.fillna(value=0);

    # In[8]:


    # results['CombinedAlpha'].describe()


    # In[9]:


    my_factor = results['CombinedAlpha']
    sectors = results['Sector']
    asset_list = results.index.levels[1].unique()
    prices = get_pricing(asset_list, start_date='2014-01-01', end_date='2018-08-29', fields='open_price')
    periods = (1, 5, 10)

    # In[10]:


    # my_factor


    # In[11]:


    factor_data = al.utils.get_clean_factor_and_forward_returns(factor=my_factor,
                                                                prices=prices,
                                                                groupby=sectors,
                                                                quantiles=2,
                                                                groupby_labels=MORNINGSTAR_SECTOR_CODES,
                                                                periods=periods)

    # In[12]:


    al.tears.create_full_tear_sheet(factor_data, by_group=True);






## factors analysis -----combined analysis
# !/usr/bin/env python
# coding: utf-8

# # Factor Analysis
#
# by Maxwell Margenot, Gil Wassermann, James Christopher Hall, and Delaney Granizo-Mackenzie.
#
# Part of the Quantopian Lecture Series:
#
# * [www.quantopian.com/lectures](https://www.quantopian.com/lectures)
# * [https://github.com/quantopian/research_public](https://github.com/quantopian/research_public)
#
# ---

# In[1]:


import numpy as np
import pandas as pd
from quantopian.research import run_pipeline
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, Returns
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.filters import QTradableStocksUS
from time import time
from quantopian.pipeline.data import morningstar
# from quantopian.pipeline.data import factset


# ## Judging a Factor with Alphalens
#
# In order to judge whether a factor is viable, we have created a package called Alphalens. Its source code is available on [github](http://github.com/quantopian/alphalens) if you want to get into the nitty-gritty of how it works. We use Alphalens to create a "tear sheet" of a factor, similar to how we use [pyfolio](http://github.com/quantopian/pyfolio) to create a tear sheet for analyzing backtests.

# In[2]:


import alphalens as al

# Alphalens takes your factor and examines how useful it is for predicting relative value through a collection of different metrics. It breaks all the stocks in your chosen universe into different quantiles based on their ranking according to your factor and analyzes the returns, information coefficient (IC), the turnover of each quantile, and provides a breakdown of returns and IC by sector.
#
# Throughout the course of this lecture we will detail how to interpret the various individual plots generated by an `Alphalens` tear sheet and include the proper call to generate the whole tear sheet at once at the end.

# ## Sector Codes
#
# These are the possible sector codes for each security, as given by Morningstar. We will use this dictionary to help categorize our results as we walk through a factor analysis so that we can break out our information by sector.

# In[3]:


MORNINGSTAR_SECTOR_CODES = {
    -1: 'Misc',
    101: 'Basic Materials',
    102: 'Consumer Cyclical',
    103: 'Financial Services',
    104: 'Real Estate',
    205: 'Consumer Defensive',
    206: 'Healthcare',
    207: 'Utilities',
    308: 'Communication Services',
    309: 'Energy',
    310: 'Industrials',
    311: 'Technology',
}

# ## Defining a universe
#
# As always, we need to define our universe. In this case we use the QTradableStocksUS, as seen in the forums [here](https://www.quantopian.com/posts/working-on-our-best-universe-yet-qtradablestocksus).

# In[4]:


universe = QTradableStocksUS()


# In[5]:


def make_factors():
    # Custom Factor 1 : Dividend Yield
    class Div_Yield(CustomFactor):
        inputs = [morningstar.valuation_ratios.dividend_yield]
        window_length = 1

        def compute(self, today, assets, out, d_y):
            out[:] = d_y[-1]

    # Custom Factor 2 : P/B Ratio
    class Price_to_Book(CustomFactor):
        inputs = [morningstar.valuation_ratios.pb_ratio]
        window_length = 1

        def compute(self, today, assets, out, p_b_r):
            out[:] = -p_b_r[-1]

    # Custom Factor 3 : Price to Trailing 12 Month Sales
    class Price_to_TTM_Sales(CustomFactor):
        inputs = [morningstar.valuation_ratios.ps_ratio]
        window_length = 1

        def compute(self, today, assets, out, ps):
            out[:] = -ps[-1]

    # Custom Factor 4 : Price to Trailing 12 Month Cashflow
    class Price_to_TTM_Cashflows(CustomFactor):
        inputs = [morningstar.valuation_ratios.pcf_ratio]
        window_length = 1

        def compute(self, today, assets, out, pcf):
            out[:] = -pcf[-1]

    class SPY_proxy(CustomFactor):
        inputs = [morningstar.valuation.market_cap]
        window_length = 1

        def compute(self, today, assets, out, mc):
            out[:] = mc[-1]

    return {
        'Price / TTM Sales': Price_to_TTM_Sales,
        'Dividen Yield': Div_Yield,
        'Price to Book': Price_to_Book,
        'SPY Proxy': SPY_proxy,
        'Price / TTM Cashflow': Price_to_TTM_Cashflows,
    }


# pipe_columns = {
#         'Price / TTM Sales':Price_to_TTM_Sales(),
#         'Dividen Yield':Div_Yield(),
#         'Price to Book':Price_to_Book(),
#         'SPY Proxy':SPY_proxy(),
#         'Price / TTM Cashflow':Price_to_TTM_Cashflows()
#     }

#     # create the pipeline for the data pull
#     Data_Pipe = Pipeline(columns = pipe_columns)


# In[6]:


# from scipy import stats

factors = make_factors()

combined_alpha = None

for name, f in factors.items():
    if combined_alpha == None:
        combined_alpha = f(mask=universe)
    else:
        combined_alpha = combined_alpha + f(mask=universe)

# In[7]:


pipe = Pipeline(
    columns={
        'CombinedAlpha': combined_alpha,
        'Sector': Sector()
    },
    screen=universe
)

start_timer = time()
results = run_pipeline(pipe, '2014-01-01', '2016-08-29')
end_timer = time()
results = results.fillna(value=0);

# In[8]:


# results['CombinedAlpha'].describe()


# In[9]:


my_factor = results['CombinedAlpha']
sectors = results['Sector']
asset_list = results.index.levels[1].unique()
prices = get_pricing(asset_list, start_date='2014-01-01', end_date='2018-08-29', fields='open_price')
periods = (1, 5, 10)

# In[10]:


# my_factor


# In[11]:


factor_data = al.utils.get_clean_factor_and_forward_returns(factor=my_factor,
                                                            prices=prices,
                                                            groupby=sectors,
                                                            quantiles=2,
                                                            groupby_labels=MORNINGSTAR_SECTOR_CODES,
                                                            periods=periods)

# In[12]:


al.tears.create_full_tear_sheet(factor_data, by_group=True);


# ## my model analysis

# In[13]:


def make_factors():
    # Custom Factor 1 : Dividend Yield
    class Div_Yield(CustomFactor):
        morningstar.operation_ratios.roe.latest > 0.1

        inputs = [morningstar.valuation_ratios.dividend_yield]
        window_length = 1

        def compute(self, today, assets, out, d_y):
            out[:] = d_y[-1]

    # Custom Factor 2 : P/B Ratio
    class Price_to_Book(CustomFactor):
        inputs = [morningstar.valuation_ratios.pb_ratio]
        window_length = 1

        def compute(self, today, assets, out, p_b_r):
            out[:] = p_b_r[-1]

    # Custom Factor 3 : Price to Trailing 12 Month Sales
    class Price_to_TTM_Sales(CustomFactor):
        inputs = [morningstar.valuation_ratios.ps_ratio]
        window_length = 1

        def compute(self, today, assets, out, ps):
            out[:] = -ps[-1]

    # Custom Factor 4 : Price to Trailing 12 Month Cashflow
    class Price_to_TTM_Cashflows(CustomFactor):
        inputs = [morningstar.valuation_ratios.pcf_ratio]
        window_length = 1

        def compute(self, today, assets, out, pcf):
            out[:] = -pcf[-1]

            # Factor 5: PE ratio

    class PE(CustomFactor):
        inputs = [morningstar.valuation_ratios.pe_ratio]
        window_length = 1

        def compute(self, today, assets, out, p_e):
            out[:] = p_e[-1]

    # PEG factor
    class PEG(CustomFactor):
        inputs = [morningstar.valuation_ratios.peg_ratio]
        window_length = 1

        def compute(self, today, assets, out, peg):
            out[:] = peg[-1]
            out[:] = np.nan_to_num(out[:])

    ##Quality
    class Quality(CustomFactor):
        inputs = [morningstar.operation_ratios.roe]
        window_length = 1

        def compute(self, today, assets, out, roe):
            out[:] = roe[-1] > 0.08

    # Volatility
    class Volatility(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):
            close = pd.DataFrame(data=close, columns=assets)
            # Since we are going to rank largest is best we need to invert the sdev.
            out[:] = 1 / np.log(close).diff().std()

    # This factor creates the synthetic S&P500
    class SPY_proxy(CustomFactor):
        inputs = [morningstar.valuation.market_cap]
        window_length = 1

        def compute(self, today, assets, out, mc):
            out[:] = mc[-1]

    return {
        'DY': Div_Yield,
        'Price/TTM sales': Price_to_TTM_Sales,
        'PEG': PEG,
        'PE': PE,
        'Price to Book': Price_to_Book,
        'Quality': Quality,
        'SPY Proxy': SPY_proxy,
        'Price / TTM Cashflow': Price_to_TTM_Cashflows,
        'Volatility': Volatility
    }


# pipe_columns = {
#         'Price / TTM Sales':Price_to_TTM_Sales(),
#         'Dividen Yield':Div_Yield(),
#         'Price to Book':Price_to_Book(),
#         'SPY Proxy':SPY_proxy(),
#         'Price / TTM Cashflow':Price_to_TTM_Cashflows()
#     }

#     # create the pipeline for the data pull
#     Data_Pipe = Pipeline(columns = pipe_columns)


# In[14]:


# from scipy import stats
roe = morningstar.operation_ratios.roe.latest
fil = roe > 0.1
mask = universe
factors = make_factors()

combined_alpha = None

for name, f in factors.items():
    if combined_alpha == None:
        combined_alpha = f(mask=mask)
    else:
        combined_alpha = combined_alpha + f(mask=mask)

# In[15]:


pipe = Pipeline(
    columns={
        'CombinedAlpha': combined_alpha,
        'Sector': Sector()
    },
    screen=universe & fil
)

start_timer = time()
results = run_pipeline(pipe, '2014-01-01', '2016-08-29')
end_timer = time()
results = results.fillna(value=0)

# In[16]:


my_factor = results['CombinedAlpha']
sectors = results['Sector']
asset_list = results.index.levels[1].unique()
prices = get_pricing(asset_list, start_date='2014-01-01', end_date='2018-08-29', fields='open_price')
periods = (1, 5, 10)

# In[17]:


factor_data = al.utils.get_clean_factor_and_forward_returns(factor=my_factor,
                                                            prices=prices,


al.tears.create_full_tear_sheet(factor_data, by_group=True);

