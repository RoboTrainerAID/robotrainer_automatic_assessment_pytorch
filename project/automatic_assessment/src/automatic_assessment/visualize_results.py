import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize(results_path: str = "/data/loocv_results.csv"):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run main.py first.")
        return

    df_plot = pd.read_csv(results_path)
    
    # Filter for Validation set
    df_val = df_plot[df_plot['Set'] == 'Validation'].copy()

    # Set the visual style
    sns.set_theme(style="whitegrid")

    # --- PLOT A: Total Error by User (Green to Red) ---
    # We use Sum of Scaled Errors to be fair across tasks, or Real? 
    # Using Real Error Sum as it's more intuitive for "how much off total"
    user_errors = df_val.groupby('User')['Absolute Error Real'].sum().sort_values()
    
    # Create color palette based on values (Green -> Red)
    norm = plt.Normalize(user_errors.min(), user_errors.max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm) # _r reverses so Red is high error
    palette = {user: sm.to_rgba(val) for user, val in user_errors.items()}

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=user_errors.index.astype(str), # Ensure categorical X axis
        y=user_errors.values, 
        hue=user_errors.index, 
        palette=palette, 
        legend=False
    )
    plt.title("Total Error by User (Validation Set) - Real Units", fontsize=16)
    plt.ylabel("Sum of Absolute Errors (Real)", fontsize=12)
    plt.xlabel("User ID", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/plot_user_errors.png")
    print("Saved figures/plot_user_errors.png")


    # --- PLOT B: Error Distribution per Task (Scaled Boxplot + Real RMSE Text) ---
    plt.figure(figsize=(14, 8))
    
    # Boxplot using SCALED errors
    ax = sns.boxplot(
        x='Task', 
        y='Absolute Error Scaled', 
        data=df_val, 
        hue='Task', 
        palette="Set3", 
        legend=False
    )
    sns.stripplot(x='Task', y='Absolute Error Scaled', data=df_val, color=".25", alpha=0.5)

    # Calculate Real RMSE per task for annotation
    tasks = df_val['Task'].unique()
    for i, task in enumerate(tasks):
        # Filter data for this task
        task_data = df_val[df_val['Task'] == task]
        # RMSE = sqrt(mean(error^2))
        rmse_real = np.sqrt(np.mean(task_data['Absolute Error Real']**2))
        
        # Position text above the boxplot (at max value + offset)
        max_val = task_data['Absolute Error Scaled'].max()
        plt.text(
            i, 
            max_val + 0.1, 
            f"RMSE:\n{rmse_real:.2f}", 
            horizontalalignment='center', 
            size='small', 
            color='black', 
            weight='semibold'
        )

    plt.title("Error Distribution per Task (Scaled Errors, Validation Set)", fontsize=16)
    plt.ylabel("Absolute Error (Scaled)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/plot_task_errors.png")
    print("Saved figures/plot_task_errors.png")


    # --- PLOT C: Predicted vs Actual (Best vs Worst Task) ---
    # Using Real values for interpretability
    task_performance = df_val.groupby('Task')['Absolute Error Real'].mean().sort_values()
    best_task = task_performance.index[0]
    worst_task = task_performance.index[-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Best Task
    sns.scatterplot(data=df_val[df_val['Task'] == best_task], x='Actual Real', y='Predicted Real', ax=axes[0], s=100, color='g')
    min_val = df_val[df_val['Task'] == best_task][['Actual Real', 'Predicted Real']].min().min()
    max_val = df_val[df_val['Task'] == best_task][['Actual Real', 'Predicted Real']].max().max()
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_title(f"Best Performer: {best_task} (Validation)", fontsize=14)

    # Subplot 2: Worst Task
    sns.scatterplot(data=df_val[df_val['Task'] == worst_task], x='Actual Real', y='Predicted Real', ax=axes[1], s=100, color='r')
    min_val = df_val[df_val['Task'] == worst_task][['Actual Real', 'Predicted Real']].min().min()
    max_val = df_val[df_val['Task'] == worst_task][['Actual Real', 'Predicted Real']].max().max()
    axes[1].plot([min_val, max_val], [min_val, max_val], 'b--', lw=2)
    axes[1].set_title(f"Worst Performer: {worst_task} (Validation)", fontsize=14)

    plt.tight_layout()
    plt.savefig("figures/plot_pred_vs_actual_best_worst.png")
    print("Saved figures/plot_pred_vs_actual_best_worst.png")


    # --- PLOT D: Overall Actual vs Predicted (Scaled, All Tasks) ---
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=df_val, 
        x='Actual Scaled', 
        y='Predicted Scaled', 
        hue='Task', 
        style='Task', 
        s=80, 
        alpha=0.7
    )
    
    # Perfect prediction line
    min_all = min(df_val['Actual Scaled'].min(), df_val['Predicted Scaled'].min())
    max_all = max(df_val['Actual Scaled'].max(), df_val['Predicted Scaled'].max())
    plt.plot([min_all, max_all], [min_all, max_all], 'k--', lw=2, label='Perfect Prediction')

    plt.title("Overall Performance: Actual vs Predicted (Scaled, Validation Set)", fontsize=16)
    plt.xlabel("Actual Value (Scaled)", fontsize=12)
    plt.ylabel("Predicted Value (Scaled)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("figures/plot_overall_actual_vs_pred.png")
    print("Saved figures/plot_overall_actual_vs_pred.png")

if __name__ == "__main__":
    visualize()