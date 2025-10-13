"""
Optimization module for statistical arbitrage strategy
Uses Optuna to optimize trading parameters
"""

import json
import logging
import os
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from backtesting_fixed import FixedBacktestingEngine

class OptunaCallBack:
    """
    Optuna callback class for logging optimization results
    """

    def __init__(self, log_file: str = "result/optimization/optimization.log.csv") -> None:
        """
        Initialize optuna callback
        
        Args:
            log_file (str): Path to log file
        """
        # Create optimization directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            filename=log_file,
            format="%(message)s",
            filemode="w",
        )
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        self.logger = logger
        
        # Log header
        self.logger.info("trial_number,correlation_threshold,max_loss_per_trade,take_profit,position_size,entry_threshold,exit_threshold,sharpe_ratio,total_return,max_drawdown,win_rate")

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Callback function for each trial
        
        Args:
            study: Optuna study object
            trial: Current trial object
        """
        if trial.value is not None:
            # Extract parameters
            params = trial.params
            sharpe_ratio = trial.value
            
            # Log trial results
            self.logger.info(
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                trial.number,
                params.get("correlation_threshold", "N/A"),
                params.get("max_loss_per_trade", "N/A"),
                params.get("take_profit", "N/A"),
                params.get("position_size", "N/A"),
                params.get("entry_threshold", "N/A"),
                params.get("exit_threshold", "N/A"),
                sharpe_ratio,
                "N/A",  # total_return - will be filled by engine
                "N/A",  # max_drawdown - will be filled by engine
                "N/A"   # win_rate - will be filled by engine
            )

class StrategyOptimizer:
    """
    Strategy optimizer using Optuna
    """
    
    def __init__(self, config_file: str = "parameter/optimization_parameter.json"):
        """
        Initialize optimizer
        
        Args:
            config_file (str): Path to optimization config file
        """
        self.config = self.load_config(config_file)
        self.callback = OptunaCallBack()
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load optimization configuration"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "random_seed": 2024,
            "no_trials": 100,
            "optimization_params": {
                "correlation_threshold": {"min": 0.01, "max": 0.5, "type": "float"},
                "max_loss_per_trade": {"min": 0.01, "max": 0.5, "type": "float"},
                "take_profit": {"min": 0.01, "max": 0.5, "type": "float"},
                "position_size": {"min": 0.01, "max": 0.5, "type": "float"},
                "entry_threshold": {"min": 0.01, "max": 0.5, "type": "float"},
                "exit_threshold": {"min": 0.01, "max": 0.5, "type": "float"}
            },
            "objective": "maximize",
            "metric": "sharpe_ratio"
        }
    
    def suggest_parameters(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dict of suggested parameters
        """
        params = {}
        opt_params = self.config["optimization_params"]
        
        for param_name, param_config in opt_params.items():
            if param_config["type"] == "float":
                step_size = param_config.get("step", 0.01)
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["min"],
                    param_config["max"],
                    step=step_size
                )
            elif param_config["type"] == "int":
                step_size = param_config.get("step", 1)
                params[param_name] = trial.suggest_int(
                    param_name,
                    int(param_config["min"]),
                    int(param_config["max"]),
                    step=step_size
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
        
        return params
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Sharpe ratio (to be maximized)
        """
        try:
            # Get suggested parameters
            params = self.suggest_parameters(trial)
            
            # Create custom config for this trial
            custom_config = self.create_custom_config(params)
            
            # Create backtesting engine
            engine = FixedBacktestingEngine(custom_config=custom_config)
            
            # Run backtesting (silent mode)
            import sys
            from io import StringIO
            
            # Capture stdout to suppress prints
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            # Disable plotting and saving for optimization
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode
            
            # Run backtesting silently (no plots)
            engine.run_fixed_arbitrage_strategy("insample", silent=True)
            results = engine.get_results()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            if results is None or len(results) == 0:
                return float('-inf')
            
            # Calculate Sharpe ratio
            returns = results['returns']
            if len(returns) == 0 or returns.std() == 0:
                return float('-inf')
            
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            
            # Log additional metrics
            total_return = (1 + returns).prod() - 1
            max_drawdown = self.calculate_max_drawdown(returns)
            win_rate = (returns > 0).mean()
            
            # Update callback with additional metrics
            self.callback.logger.info(
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                trial.number,
                params.get("correlation_threshold", "N/A"),
                params.get("max_loss_per_trade", "N/A"),
                params.get("take_profit", "N/A"),
                params.get("position_size", "N/A"),
                params.get("entry_threshold", "N/A"),
                params.get("exit_threshold", "N/A"),
                sharpe_ratio,
                total_return,
                max_drawdown,
                win_rate
            )
            
            return sharpe_ratio
            
        except Exception as e:
            # print(f"Error in trial {trial.number}: {e}")
            return float('-inf')
    
    def create_custom_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create custom configuration for backtesting
        
        Args:
            params: Optimization parameters
            
        Returns:
            Custom configuration dictionary
        """
        # Load base config
        with open("parameter/in_sample.json", 'r', encoding='utf-8') as f:
            base_config = json.load(f)
        
        # Update with optimization parameters
        for param_name, param_value in params.items():
            if param_name in base_config:
                base_config[param_name] = param_value
        
        return base_config
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _progress_callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Progress callback to show optimization progress"""
        if trial.number % 10 == 0 or trial.number == 0:
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            total = len(study.trials)
            # print(f"📊 Progress: {completed}/{total} trials completed ({(completed/total)*100:.1f}%)")
            
            if completed > 0:
                best_value = study.best_value
                # print(f"🏆 Best Sharpe ratio so far: {best_value:.4f}")
    
    def optimize(self) -> optuna.study.Study:
        """
        Run optimization
        
        Returns:
            Optuna study object with results
        """
        # print("🚀 Starting parameter optimization...")
        # print(f"📊 Number of trials: {self.config['no_trials']}")
        # print(f"🎯 Objective: {self.config['objective']} {self.config['metric']}")
        # print(f"🔧 Parameters to optimize: {list(self.config['optimization_params'].keys())}")
        # print("⏳ Running optimization (silent mode)...")
        
        # Create study
        study = optuna.create_study(
            sampler=TPESampler(seed=self.config["random_seed"]),
            direction=self.config["objective"],
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Run optimization with progress callback
        study.optimize(
            self.objective,
            n_trials=self.config["no_trials"],
            callbacks=[self.callback, self._progress_callback]
        )
        
        # Print results (disabled to avoid table display)
        # self.print_results(study)
        
        # Save best parameters
        self.save_best_parameters(study)
        
        return study
    
    def print_results(self, study: optuna.study.Study) -> None:
        """Print optimization results"""
        print("\n" + "="*60)
        print("🎯 OPTIMIZATION RESULTS")
        print("="*60)
        
        best_trial = study.best_trial
        print(f"✅ Best trial: {best_trial.number}")
        print(f"📈 Best Sharpe ratio: {best_trial.value:.4f}")
        
        print(f"\n🔧 Best parameters:")
        for param, value in best_trial.params.items():
            print(f"  • {param}: {value:.4f}")
        
        print(f"\n📊 Optimization statistics:")
        print(f"  • Total trials: {len(study.trials)}")
        print(f"  • Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  • Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
        # Show top 5 trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        completed_trials.sort(key=lambda x: x.value, reverse=True)
        
        print(f"\n🏆 Top 5 trials:")
        for i, trial in enumerate(completed_trials[:5]):
            print(f"  {i+1}. Trial {trial.number}: Sharpe = {trial.value:.4f}")
    
    def save_best_parameters(self, study: optuna.study.Study) -> None:
        """Save best parameters to file"""
        best_params = study.best_params
        best_value = study.best_value
        
        result = {
            "best_sharpe_ratio": best_value,
            "best_parameters": best_params,
            "optimization_config": self.config,
            "total_trials": len(study.trials)
        }
        
        # Save to JSON
        with open("result/optimization/best_parameters.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # print(f"\n💾 Best parameters saved to: result/optimization/best_parameters.json")

def main():
    """Main function"""
    try:
        # Create optimizer
        optimizer = StrategyOptimizer()
        
        # Run optimization
        study = optimizer.optimize()
        
        print("\n✅ Optimization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        raise

if __name__ == "__main__":
    main()
