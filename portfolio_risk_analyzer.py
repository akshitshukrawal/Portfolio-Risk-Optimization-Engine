"""
Portfolio Risk Management & Optimization System
================================================

A comprehensive system that includes:
- PortfolioRiskAnalyzer: Analyzes risk for a single, predefined portfolio.
- BacktestingEngine: Validates VaR models against historical data.
- PortfolioOptimizer: Finds the optimal portfolio from a universe of assets using Monte Carlo simulation.

Author: Portfolio Risk Management Project
Date: 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

#==============================================================================
# CLASS 1: ANALYZER FOR A SINGLE, PREDEFINED PORTFOLIO
#==============================================================================
class PortfolioRiskAnalyzer:
    """
    Analyzes the risk of a portfolio with predefined weights.
    """
    def __init__(self, symbols, weights, start_date, end_date):
        self.symbols = symbols
        self.weights = np.array(weights)
        self.start_date = start_date
        self.end_date = end_date
        
        if abs(self.weights.sum() - 1.0) > 1e-6:
            raise ValueError("Portfolio weights must sum to 1.0")

        self._download_data()
        self._calculate_returns()

    def _download_data(self):
        print("Downloading data for risk analysis...")
        self.data = yf.download(self.symbols, start=self.start_date, end=self.end_date, progress=False)['Adj Close']
        self.data = self.data.dropna()
        print(f"✅ Downloaded data for {len(self.symbols)} symbols: {len(self.data)} observations")

    def _calculate_returns(self):
        self.returns = self.data.pct_change().dropna()
        self.portfolio_returns = (self.returns * self.weights).sum(axis=1)
        print(f"Returns calculated: {len(self.portfolio_returns)} daily returns")

    def calculate_var(self, confidence_level=0.95, method='historical'):
        if method == 'historical':
            return np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
        
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()

        if method == 'parametric':
            z_score = stats.norm.ppf(1 - confidence_level)
            return mu + z_score * sigma
        
        if method == 'cornish_fisher':
            skew = stats.skew(self.portfolio_returns)
            kurt = stats.kurtosis(self.portfolio_returns)
            z_norm = stats.norm.ppf(1 - confidence_level)
            z_cf = (z_norm + (z_norm**2 - 1) * skew / 6 + (z_norm**3 - 3*z_norm) * kurt / 24 - (2*z_norm**3 - 5*z_norm) * (skew**2) / 36)
            return mu + z_cf * sigma
            
        elif method == 'monte_carlo':
            mean_returns = self.returns.mean()
            cov_matrix = self.returns.cov()
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, 10000)
            portfolio_sim_returns = np.dot(random_returns, self.weights)
            return np.percentile(portfolio_sim_returns, (1 - confidence_level) * 100)
        
        raise ValueError("Method must be 'historical', 'parametric', 'cornish_fisher', or 'monte_carlo'")

    def calculate_expected_shortfall(self, confidence_level=0.95, method='historical'):
        var = self.calculate_var(confidence_level, method)
        tail_losses = self.portfolio_returns[self.portfolio_returns <= var]
        return tail_losses.mean() if not tail_losses.empty else var

    def plot_correlation_heatmap(self, save_path=None):
        plt.style.use('seaborn-v0_8-whitegrid')
        corr_matrix = self.returns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Asset Correlation Heatmap')
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_return_distribution(self, save_path=None):
        plt.style.use('seaborn-v0_8-whitegrid')
        var_95 = self.calculate_var(0.95, 'historical')
        es_95 = self.calculate_expected_shortfall(0.95, 'historical')
        plt.figure(figsize=(12, 7))
        sns.histplot(self.portfolio_returns, bins=50, kde=True, color='royalblue')
        plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'Historical VaR (95%): {var_95:.2%}')
        plt.axvline(es_95, color='purple', linestyle='--', linewidth=2, label=f'Expected Shortfall (95%): {es_95:.2%}')
        plt.title('Portfolio Daily Return Distribution')
        plt.xlabel('Daily Returns'); plt.ylabel('Frequency'); plt.legend()
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()

#==============================================================================
# CLASS 2: BACKTESTING ENGINE
#==============================================================================
class BacktestingEngine:
    """
    Engine for backtesting VaR models on historical data.
    """
    def __init__(self, analyzer: PortfolioRiskAnalyzer):
        self.analyzer = analyzer

    def run_full_backtest(self, method='historical', confidence_level=0.95, rolling_window=252):
        returns = self.analyzer.portfolio_returns
        var_series = returns.rolling(window=rolling_window).apply(
            lambda x: np.percentile(x, (1 - confidence_level) * 100), raw=True
        ).dropna()
        
        actual_returns = self.analyzer.portfolio_returns.loc[var_series.index]
        violations = actual_returns < var_series
        n_obs, n_violations = len(var_series), violations.sum()
        
        p = 1 - confidence_level
        pi = n_violations / n_obs
        log_likelihood_unrestricted = n_violations * np.log(pi) + (n_obs - n_violations) * np.log(1 - pi)
        log_likelihood_restricted = n_violations * np.log(p) + (n_obs - n_violations) * np.log(1 - p)
        kupiec_stat = -2 * (log_likelihood_restricted - log_likelihood_unrestricted)
        p_value = 1 - stats.chi2.cdf(kupiec_stat, df=1)

        summary = { 'n_observations': n_obs, 'n_violations': n_violations, 'violation_rate': pi,
                    'expected_rate': p, 'kupiec_statistic': kupiec_stat, 'kupiec_p_value': p_value }
        
        backtest_df = pd.DataFrame({'var_estimate': var_series, 'actual_return': actual_returns, 'violation': violations})
        return summary, backtest_df

#==============================================================================
# CLASS 3: PORTFOLIO OPTIMIZER
#==============================================================================
class PortfolioOptimizer:
    """
    Finds the optimal portfolio from a universe of assets using Monte Carlo simulation.
    """
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self._download_data()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        self.results_df = None

    def _download_data(self):
        print("Downloading data for portfolio optimization...")
        self.data = yf.download(self.symbols, start=self.start_date, end=self.end_date, progress=False)['Adj Close']
        self.data = self.data.dropna()
        self.returns = self.data.pct_change().dropna()
        print(f"✅ Optimizer data downloaded for {len(self.symbols)} symbols.")

    def run_simulation(self, n_portfolios=20000, risk_free_rate=0.02):
        print(f"Running Monte Carlo simulation for {n_portfolios} portfolios...")
        results = np.zeros((n_portfolios, 4 + len(self.symbols)))
        
        for i in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(len(self.symbols))
            weights /= np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(self.mean_returns * weights) * 252
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
            
            # Store results
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_stddev
            results[i, 2] = (portfolio_return - risk_free_rate) / portfolio_stddev # Sharpe Ratio
            
            # Calculate Historical VaR for this portfolio
            portfolio_daily_returns = (self.returns * weights).sum(axis=1)
            results[i, 3] = np.percentile(portfolio_daily_returns, 5) # 95% VaR
            
            for j in range(len(weights)):
                results[i, j + 4] = weights[j]

        cols = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'var_95'] + self.symbols
        self.results_df = pd.DataFrame(results, columns=cols)
        print("✅ Simulation complete.")
        return self.results_df

    def get_optimal_portfolios(self):
        if self.results_df is None:
            raise Exception("Please run the simulation first.")
        
        # Max Sharpe Ratio Portfolio
        max_sharpe_portfolio = self.results_df.iloc[self.results_df['sharpe_ratio'].idxmax()]
        
        # Minimum Volatility Portfolio
        min_vol_portfolio = self.results_df.iloc[self.results_df['annual_volatility'].idxmin()]

        # Minimum VaR Portfolio (safest from a tail-risk perspective)
        min_var_portfolio = self.results_df.iloc[self.results_df['var_95'].idxmax()] # idxmax because VaR is negative

        return max_sharpe_portfolio, min_vol_portfolio, min_var_portfolio

    def plot_efficient_frontier(self, save_path=None):
        if self.results_df is None:
            raise Exception("Please run the simulation first.")

        max_sharpe, min_vol, min_var = self.get_optimal_portfolios()

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of all simulated portfolios
        plt.scatter(self.results_df['annual_volatility'], self.results_df['annual_return'], c=self.results_df['sharpe_ratio'], cmap='viridis', marker='o', s=10, alpha=0.7)
        plt.colorbar(label='Sharpe Ratio')
        
        # Highlight optimal portfolios
        plt.scatter(max_sharpe['annual_volatility'], max_sharpe['annual_return'], marker='*', color='red', s=200, label='Max Sharpe Ratio')
        plt.scatter(min_vol['annual_volatility'], min_vol['annual_return'], marker='*', color='blue', s=200, label='Min Volatility')
        plt.scatter(min_var['annual_volatility'], min_var['annual_return'], marker='*', color='purple', s=200, label='Min VaR (95%)')
        
        plt.title('Portfolio Optimization - Efficient Frontier')
        plt.xlabel('Annual Volatility (Risk)')
        plt.ylabel('Annual Return')
        plt.legend(labelspacing=0.8)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()