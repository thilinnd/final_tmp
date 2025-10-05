import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from utils.helper import generate_periods_df, run_backtest_for_periods
from utils.calculate_metrics import calculate_shapre_and_mdd
from data.get_data import *
from optuna.pruners import MedianPruner
import warnings
import logging

# Set up logging to suppress Optuna's internal messages by default
# Change to optuna.logging.DEBUG to see detailed logs if needed
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress to warnings only (INFO and below are hidden)
# Redirect Optuna logs to Python's logging module at DEBUG level
optuna.logging.enable_propagation()  # Allow Optuna logs to propagate to Python's logging
logging.basicConfig(level=logging.WARNING)  # Set default logging level to WARNING (hides DEBUG and INFO)

pd.set_option('future.no_silent_downcasting', True)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory (project2)
os.chdir(script_dir)

# Suppress the specific Optuna warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna.distributions"
)

# Set random seeds for reproducibility
SEED = 42  # You can choose any integer
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)  # For hash-based operations

vn30_stocks = pd.read_csv('data/vn30_stocks.csv', index_col=0, parse_dates=True)
# Load data once at the start
etf_list = ['FUEVFVND', 'FUESSVFL', 'E1VFVN30', 'FUEVN100']
start_date = '2021-06-01'
end_date = '2025-01-01'
periods_df = generate_periods_df(
    vn30_stocks,
    start_date,
    end_date,
    window=80,
)

def objective(trial):
    # Suggest parameters to tune
    estimation_window = trial.suggest_int("estimation_window", 40, 80, step=10)
    min_trading_days_fraction = trial.suggest_float(
        "min_trading_days", 0.5, 0.8, step=0.15
    )
    min_trading_days = int(min_trading_days_fraction * estimation_window)  # Convert fraction to days
    max_clusters = trial.suggest_int("max_clusters", 8, 10, step=2)
    top_stocks = trial.suggest_int("top_stocks", 3, 9, step=3)
    correlation_threshold = trial.suggest_float(
        "correlation_threshold", 0.3, 0.6, step=0.3
    )

    ou_window = estimation_window  
    tier = trial.suggest_categorical("tier", [1, 2, 3, 4])

    first_allocation = trial.suggest_float("first_allocation", 0.4, 0.7, step=0.15)
    adding_allocation = trial.suggest_float("adding_allocation", 0.2, 0.3, step=0.1)
    
    # Run backtest with suggested parameters
    combined_returns_df, _, _ = run_backtest_for_periods(
        periods_df=periods_df,
        futures="VN30F1M",
        etf_list=etf_list,
        etf_included=False,
        estimation_window=estimation_window,
        min_trading_days=min_trading_days_fraction * estimation_window,  # Convert fraction to days
        max_clusters=max_clusters,
        top_stocks=top_stocks,
        correlation_threshold=correlation_threshold,
        tier=tier,
        first_allocation=first_allocation,
        adding_allocation=adding_allocation,
        use_existing_data=True
    )

    # Split into train and test sets
    train_set = combined_returns_df[combined_returns_df.index < "2024-01-01"]
    test_set = combined_returns_df[combined_returns_df.index >= "2024-01-01"]

    # Calculate metrics for train set (optimization targets)
    return_train, sharpe_train, max_drawdown_train = calculate_shapre_and_mdd(train_set, risk_free_rate=0.05)

    # Calculate metrics for test set (for reporting, not optimization)
    return_test, sharpe_test, max_drawdown_test = calculate_shapre_and_mdd(test_set, risk_free_rate=0.05)

    # Store test set metrics for later analysis
    trial.set_user_attr("sharpe_test", sharpe_test)
    trial.set_user_attr("return_test", return_test)
    trial.set_user_attr("max_drawdown_test", max_drawdown_test)

    # Return tuple for multi-objective optimization (maximize return, maximize Sharpe, minimize drawdown)
    return return_train, sharpe_train, max_drawdown_train


# Create study
study = optuna.create_study(
    directions=["maximize", "maximize", "minimize"],  # Maximize return, maximize Sharpe, minimize drawdown
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="vn30_arbitrage_tuning",
    load_if_exists=True,
    pruner=MedianPruner()
)

# Parameters
total_trials = 50
batch_size = 10
n_batches = total_trials // batch_size  # 5 batches

# CSV file name in data folder
csv_file = os.path.join(script_dir, "data", "optuna_trials_final_real.csv")

# Ensure the data directory exists
os.makedirs(os.path.dirname(csv_file), exist_ok=True)


# Run trials in batches
for batch in range(n_batches):
    # Run one batch of trials
    study.optimize(objective, n_trials=batch_size)
    
    # Get all trials up to this point
    trials_df = study.trials_dataframe()
    
    # Rename columns to match your variable names
    column_mapping = {
        'values_0': 'Return_train',
        'values_1': 'Sharpe_train',
        'values_2': 'Drawdown_train',
        'user_attrs_sharpe_test': 'Sharpe_test',
        'user_attrs_return_test': 'Return_test',
        'user_attrs_max_drawdown_test': 'Drawdown_test',
    }
    
    # Rename the columns
    trials_df.rename(columns=column_mapping, inplace=True)
    
    # Keep only relevant columns
    param_columns = [col for col in trials_df.columns if col.startswith('params_')]
    relevant_columns = ['number', 'Return_train', 'Sharpe_train', 'Drawdown_train', 
                       'Return_test', 'Sharpe_test', 'Drawdown_test'] + param_columns
    trials_df = trials_df[relevant_columns]
    
    # Write or append to CSV (still useful for post-analysis)
    if batch == 0:
        trials_df.to_csv(csv_file, mode='w', index=False)
    else:
        # Only append the new trials (last batch_size trials)
        new_trials_df = trials_df.tail(batch_size)
        new_trials_df.to_csv(csv_file, mode='a', header=False, index=False)
    
    # # Print batch progress (commented out for concise logging)
    # print(f"Batch {batch+1}/{n_batches} completed ({(batch+1)*batch_size}/{total_trials} trials)")
    # print(f"Latest batch results:")
    # for _, row in trials_df.tail(batch_size).iterrows():
    #     print(f"Trial {int(row['number'])}: "
    #           f"Return Train: {row['Return_train']}, "
    #           f"Sharpe Train: {row['Sharpe_train']}, "
    #           f"Drawdown Train: {row['Drawdown_train']}, "
    #           f"Return Test: {row['Return_test']}, "
    #           f"Sharpe Test: {row['Sharpe_test']}, "
    #           f"Drawdown Test: {row['Drawdown_test']}")
    # print("-" * 50)

# Get and print best trial results (note: with multi-objective, this is one of the Pareto optimal solutions)
best_trial = study.best_trials[0]  # Get the first Pareto optimal trial
best_params = best_trial.params
best_return_train = best_trial.values[0]
best_sharpe_train = best_trial.values[1]
best_drawdown_train = best_trial.values[2]
best_return_test = best_trial.user_attrs["return_test"]
best_sharpe_test = best_trial.user_attrs["sharpe_test"]
best_drawdown_test = best_trial.user_attrs["max_drawdown_test"]

# Only print the final best results
print("\nFinal Best Results (One of the Pareto Optimal Solutions):")
print(f"Best Train Return: {best_return_train}")
print(f"Best Train Sharpe Ratio: {best_sharpe_train}")
print(f"Best Train Max Drawdown: {best_drawdown_train}")
print(f"Test Return: {best_return_test}")
print(f"Test Sharpe Ratio: {best_sharpe_test}")
print(f"Test Max Drawdown: {best_drawdown_test}")
print(f"Best Parameters: {best_params}")

# # Visualize parameter importances (for Return only as an example, since it's multi-objective)
# optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0]).show()

# # Extract Return, Sharpe Ratios, and trial numbers for visualization
# returns = [trial.values[0] for trial in study.trials if trial.values is not None]
# sharpe_ratios = [trial.values[1] for trial in study.trials if trial.values is not None]
# trial_numbers = list(range(len(sharpe_ratios)))

# # Compute the best Return and Sharpe Ratio seen so far for each trial
# best_return_so_far = []
# best_sharpe_so_far = []
# current_best_return = float("-inf")
# current_best_sharpe = float("-inf")
# for ret, sharpe in zip(returns, sharpe_ratios):
#     current_best_return = max(current_best_return, ret)
#     current_best_sharpe = max(current_best_sharpe, sharpe)
#     best_return_so_far.append(current_best_return)
#     best_sharpe_so_far.append(current_best_sharpe)

# # Create the plot for Return
# plt.figure(figsize=(10, 6))
# plt.scatter(trial_numbers, returns, color="blue", alpha=0.5, label="Return per Trial")
# plt.plot(trial_numbers, best_return_so_far, color="red", label="Best Return")
# plt.xlabel("Trial")
# plt.ylabel("Return (Objective)")
# plt.title("Return Over Trials")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Create the plot for Sharpe Ratio
# plt.figure(figsize=(10, 6))
# plt.scatter(trial_numbers, sharpe_ratios, color="blue", alpha=0.5, label="Sharpe Ratio per Trial")
# plt.plot(trial_numbers, best_sharpe_so_far, color="red", label="Best Sharpe Ratio")
# plt.xlabel("Trial")
# plt.ylabel("Sharpe Ratio (Objective)")
# plt.title("Sharpe Ratio Over Trials")
# plt.legend()
# plt.grid(True)
# plt.show()