import pandas as pd
from pathlib import Path
pd.set_option('display.width', None)
import numpy as np
import matplotlib.pyplot as plt


class SMAStrategy:
    def __init__(self, sma_small:int, sma_large:int, filepath):
        self.sma_small = sma_small
        self.sma_large = sma_large
        self.filename = filepath
        self.symbol = Path(self.filename).stem
        self.data = self.get_data()
        self.position_data = self.get_position()[0]
        self.shorts = self.get_position()[1]
        self.calc_returns_data = self.calc_returns()
        self.plot_results = self.plot_results()

    def get_data(self):
        data = pd.read_csv(self.filename, parse_dates=[0], usecols=[0, 2, 3, 4, 5])
        data.columns = ['datetime', 'open', 'high', 'low', 'close']
        data[f"sma_{self.sma_small}"] = data['close'].rolling(window=self.sma_small).mean()
        data[f"sma_{self.sma_large}"] = data['close'].rolling(window=self.sma_large).mean()
        data['delta'] = data[f"sma_{self.sma_small}"] - data[f"sma_{self.sma_large}"]
        data['delta_prev'] = data['delta'].shift(1)
        data['date']  = data['datetime'].dt.date
        data.set_index(['date'], inplace=True)
        return data

    def get_position(self, shorts:bool=False):
        data = self.data
        shorts = shorts
        data['position'] = np.nan
        data['position'] = np.where((data["delta"] >= 0)&\
                                    (data["delta_prev"]< 0),
                                    1, data['position'])
        if shorts:
            data['position'] = np.where((data["delta"] < 0)&\
                                    (data["delta_prev"]>= 0),
                                    -1, data['position'])
        else:
            data['position'] = np.where((data["delta"] < 0 )&\
                                    (data["delta_prev"]>= 0),
                                    0, data['position'])
        data['position'] = data['position'].ffill().fillna(0)
        return data, shorts

    def calc_returns(self):
        data = self.get_position()[0]
        data['returns'] = data['close'].div(data['close'].shift(1))
        data['log_returns'] = np.log(data['returns'])
        data['strategy_returns'] = data['returns'] * data['position'].shift(1)
        data['strategy_log_returns'] = data['log_returns'] * data['position'].shift(1)
        data['cum_returns'] = data['log_returns'].cumsum().apply(np.exp)
        data['strategy_cum_returns'] = data['strategy_log_returns'].cumsum().apply(np.exp)
        data['peak'] = data['cum_returns'].cummax()
        data['strategy_peak'] = data['strategy_cum_returns'].cummax()
        return data

    def get_strategy_stats(self, log_returns: pd.Series, risk_free_rate: float  = 0.083925 ):
        stats = {}
        stats['tot_returns'] = np.exp(log_returns.sum()) - 1
        stats['annual_returns'] = np.exp(log_returns.mean() * 252) - 1
        stats['annual_volatility'] = log_returns.std() * np.sqrt(252)
        annualised_downside = log_returns.loc[log_returns < 0 ].std() * np.sqrt(252)
        stats['sortino_ratio'] = (stats['annual_returns'] - risk_free_rate) / annualised_downside
        stats['sharpe_ratio'] = (stats['annual_returns'] - risk_free_rate) / stats['annual_volatility']
        cum_returns = log_returns.cumsum()
        peak = cum_returns.cummax()
        draw_down = peak - cum_returns
        max_idx = draw_down.argmax()
        stats['max_dd'] = 1 - np.exp(cum_returns.iloc[max_idx]) / np.exp(peak.iloc[max_idx])
        strat_dd = draw_down[draw_down==0]
        strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
        strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
        strat_dd_days = np.hstack([strat_dd_days, (draw_down.index[-1] - strat_dd.index[-1]).days ])
        stats['max_dd_duration'] = strat_dd_days.max()
        return {k: np.round(v, 4) if type(v) == float else v for k, v in stats.items()}

    def plot_results(self):
        df_plot = self.calc_returns_data
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df_plot['strategy_cum_returns']  * 100, label=f"{self.sma_small}_{self.sma_large} crossover ")
        ax.plot(df_plot['cum_returns'] * 100, label="Buy and Hold")
        ax.set_title(F"{self.symbol} cumulative returns for {self.sma_small}_{self.sma_large}crossover.png")
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns %")
        ax.legend()
        return plt.show()
    
    
data = SMAStrategy(sma_small=4, sma_large=9, filepath = "KCB.csv")
stats = pd.DataFrame(data.get_strategy_stats(data.calc_returns_data['log_returns']), index=['Buy and hold'])
stats = pd.concat([stats, pd.DataFrame(data.get_strategy_stats(data.calc_returns_data['strategy_log_returns']),
                                        index=[f"{data.sma_small}_{data.sma_large} crossover"])])
print(stats)
print(data.plot_results)