"""
Portfolio Risk Management & Optimization - Main Execution Script
================================================================

This script demonstrates the complete system:
1. Analyzes the risk of a single, predefined portfolio.
2. Runs a Monte Carlo simulation to find the optimal portfolio from a universe of assets.

Usage: python main.py
"""
import os
import pandas as pd
from portfolio_risk_analyzer import PortfolioRiskAnalyzer, BacktestingEngine, PortfolioOptimizer

def main():
    """Main execution function"""
    print("Portfolio Risk Management & Optimization System")
    print("=" * 60)

    # --- Create a directory for saving plots ---
    output_dir = "risk_analysis_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #==========================================================================
    # PART 1: RISK ANALYSIS OF A PREDEFINED PORTFOLIO
    #==========================================================================
    print("\nPART 1: Analyzing a Predefined 'Tech Portfolio'")
    print("-" * 60)
    
    try:
        # --- Define a single portfolio for analysis ---
        tech_portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
        tech_portfolio_weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        start_date = '2015-08-01'
        end_date = '2025-08-01'

        # --- Initialize and run the analysis ---
        analyzer = PortfolioRiskAnalyzer(tech_portfolio_symbols, tech_portfolio_weights, start_date, end_date)
        
        # --- Display Key Metrics ---
        daily_returns = analyzer.portfolio_returns
        print(f"\nAnnual Return: {daily_returns.mean() * 252:.2%}")
        print(f"Annual Volatility: {daily_returns.std() * 252**0.5:.2%}")

        print("\nValue at Risk (95% confidence):")
        var_hist = analyzer.calculate_var(0.95, 'historical')
        var_cf = analyzer.calculate_var(0.95, 'cornish_fisher')
        print(f"  Historical VaR: {var_hist:.2%}")
        print(f"  Cornish-Fisher VaR: {var_cf:.2%}")
        var_hist = analyzer.calculate_var(0.95, 'historical')
        var_cf = analyzer.calculate_var(0.95, 'cornish_fisher')
        print(f"  Historical VaR: {var_hist:.2%}")
        print(f"  Cornish-Fisher VaR: {var_cf:.2%}")

        # --- Generate and Save Visualizations for the single portfolio ---
        analyzer.plot_return_distribution(save_path=f"{output_dir}/tech_portfolio_returns_dist.png")
        print(f"\n✅ Risk analysis for predefined portfolio complete. Plots saved to '{output_dir}/'")

    except Exception as e:
        print(f"❌ Error in Part 1: {e}")


    #==========================================================================
    # PART 2: FINDING THE OPTIMAL PORTFOLIO
    #==========================================================================
    print("\n\nPART 2: Finding the Optimal Portfolio using Monte Carlo")
    print("-" * 60)

    try:
        # --- Define a universe of assets to build a portfolio from ---
        asset_universe = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM', 'GOOGL', 'META']
        opt_start_date = '2021-08-01'
        opt_end_date = '2025-08-01'
        
        # --- Initialize and run the optimizer ---
        optimizer = PortfolioOptimizer(asset_universe, opt_start_date, opt_end_date)
        optimizer.run_simulation(n_portfolios=20000)

        # --- Get and display the optimal portfolios ---
        max_sharpe, min_vol, min_var = optimizer.get_optimal_portfolios()

        print("\n--- Optimal Portfolio: Max Sharpe Ratio ---")
        print(f"Annual Return: {max_sharpe['annual_return']:.2%}")
        print(f"Annual Volatility: {max_sharpe['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.2f}")
        print(f"95% VaR: {max_sharpe['var_95']:.2%}")
        print("Optimal Weights:")
        print(max_sharpe[asset_universe].round(4))

        print("\n--- Optimal Portfolio: Minimum VaR (Safest) ---")
        print(f"Annual Return: {min_var['annual_return']:.2%}")
        print(f"Annual Volatility: {min_var['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {min_var['sharpe_ratio']:.2f}")
        print(f"95% VaR: {min_var['var_95']:.2%}")
        print("Optimal Weights:")
        print(min_var[asset_universe].round(4))

        # --- Plot and save the efficient frontier ---
        print("\nDisplaying the Efficient Frontier plot...")
        optimizer.plot_efficient_frontier(save_path=f"{output_dir}/efficient_frontier.png")
        print(f"\n✅ Portfolio optimization complete. Plot saved to '{output_dir}/'")

    except Exception as e:
        print(f"❌ Error in Part 2: {e}")

    print("\n\n✅ Full Process Finished!")

if __name__ == "__main__":
    main()
