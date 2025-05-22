"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        
        # 簡單但有效的動量策略
        for i in range(self.lookback, len(self.price)):
            current_date = self.price.index[i]
            
            # 使用更長的lookback period來計算動量
            lookback_period = min(126, i)  # 6個月或可用的最大天數
            start_idx = i - lookback_period
            
            # 計算各sector的總回報
            total_returns = (self.price.iloc[i][assets] / self.price.iloc[start_idx][assets] - 1)
            
            # 計算短期動量 (1個月)
            short_lookback = min(21, i)
            short_start_idx = i - short_lookback
            short_returns = (self.price.iloc[i][assets] / self.price.iloc[short_start_idx][assets] - 1)
            
            # 計算波動率
            hist_returns = self.returns.iloc[start_idx:i][assets]
            volatility = hist_returns.std() * np.sqrt(252)
            
            # 風險調整後的動量分數
            risk_adj_returns = total_returns / volatility
            risk_adj_short = short_returns / volatility
            
            # 綜合分數：長期動量 + 短期動量
            composite_score = 0.6 * risk_adj_returns + 0.4 * risk_adj_short
            
            # 排序所有sector
            sector_ranks = composite_score.rank(ascending=False)
            
            # 集中投資策略：只投資前3名的sector
            weights = np.zeros(len(assets))
            
            # 找出前3名
            top_3_mask = sector_ranks <= 3
            top3_sectors = assets[top_3_mask]
            
            if len(top3_sectors) >= 3:
                # 權重分配：第1名 50%, 第2名 30%, 第3名 20%
                sorted_top3 = composite_score[top3_sectors].sort_values(ascending=False)
                
                for j, asset in enumerate(assets):
                    if asset == sorted_top3.index[0]:  # 第1名
                        weights[j] = 0.50
                    elif asset == sorted_top3.index[1]:  # 第2名
                        weights[j] = 0.30
                    elif asset == sorted_top3.index[2]:  # 第3名
                        weights[j] = 0.20
            else:
                # 如果數據不足，平均分配給前3名
                for j, asset in enumerate(assets):
                    if top_3_mask[j]:
                        weights[j] = 1.0 / top_3_mask.sum()
            
            # 安全檢查：確保權重和為1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # 極端情況：等權重分配
                weights = np.ones(len(assets)) / len(assets)
            
            # 分配權重到組合
            for j, asset in enumerate(assets):
                self.portfolio_weights.loc[current_date, asset] = weights[j]

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Assignment Judge

The following functions will help check your solution.
"""


class AssignmentJudge:
    def __init__(self):
        self.mp = MyPortfolio(df, "SPY").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY").get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    if args.score:
        if ("one" in args.score) or ("spy" in args.score):
            if "one" in args.score:
                judge.check_sharp_ratio_greater_than_one()
            if "spy" in args.score:
                judge.check_sharp_ratio_greater_than_spy()
        elif "all" in args.score:
            print(f"==> total Score = {judge.check_all_answer()} <==")

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)

    if args.cumulative:
        if "mp" in args.cumulative:
            judge.cumulative_product(df)
        if "bmp" in args.cumulative:
            judge.cumulative_product(Bdf)