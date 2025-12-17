import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_loocv_results(results_path: str = "/data/loocv_results.csv"):
    """
    Generates plots based on the Leave-One-Out Cross-Validation results CSV.
    """
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

def visualize_tuning_convergence(cv_results: dict, n_iter: int, model_type: str):
    """
    Visualizes the convergence of the hyperparameter tuning process.
    Plots individual trial scores and the cumulative best score.
    """
    # Extract results. Scikit-learn returns negative MSE, so we negate it back to positive
    mean_scores = -cv_results['mean_test_score'] # The "Chaos" (dots)
    
    # Calculate 'Best So Far' (Cumulative Minimum) to visualize the "Elbow"
    # np.minimum.accumulate creates a stepping down line
    best_so_far = np.minimum.accumulate(mean_scores)
    
    iterations = range(1, len(mean_scores) + 1)

    plt.figure(figsize=(10, 6))
    
    # Plot 1: The individual trials (Grey dots = Chaos/Exploration)
    plt.scatter(iterations, mean_scores, color='grey', alpha=0.6, label='Individual Trial (MSE)')
    
    # Plot 2: The Best So Far (Red Line = Convergence/Elbow)
    plt.plot(iterations, best_so_far, color='red', linewidth=2, label='Best Score So Far')
    
    # Formatting to match your description
    plt.title(f'Hyperparameter Tuning Convergence ({model_type.upper()})')
    plt.xlabel('Iteration Number')
    plt.ylabel('Mean Squared Error (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add annotated zones based on your description (Optional visual guide)
    if n_iter >= 50:
        plt.axvline(x=15, color='blue', linestyle=':', alpha=0.5)
        plt.text(7.5, max(best_so_far), 'Exploration\n(Chaos)', ha='center', fontsize=9, color='blue')
        
        plt.axvline(x=40, color='blue', linestyle=':', alpha=0.5)
        plt.text(27.5, max(best_so_far), 'Improvement', ha='center', fontsize=9, color='blue')
        
        plt.text(n_iter, max(best_so_far), 'Fine Tuning\n(Elbow)', ha='right', fontsize=9, color='blue')

    plt.tight_layout()
    plt.savefig("figures/plot_tuning_convergence_" + model_type + ".png")
    print("Saved figures/plot_tuning_convergence_" + model_type + ".png")

def visualize_mlp_loss(mlp_model, model_name="MLP"):
    """
    Plots the training loss curve for an MLP model.
    """
    if not hasattr(mlp_model, 'loss_curve_'):
        print("Model does not have a loss_curve_. Is it an MLP trained with a solver that supports it? This is expected if using 'lbfgs' solver.")
        return

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(mlp_model.loss_curve_, label='Training Loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Check for validation scores (available if early_stopping=True or validation_fraction > 0)
    if hasattr(mlp_model, 'validation_scores_') and mlp_model.validation_scores_ is not None:
        # Plot validation score on secondary axis as it is usually R2 (different scale/direction than Loss)
        ax2 = ax1.twinx()
        ax2.plot(mlp_model.validation_scores_, label='Validation Score', color='orange', linestyle='--')
        ax2.set_ylabel('Validation Score')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax1.legend()

    # Note: Sklearn MLP doesn't always store validation scores automatically 
    # unless you use early_stopping=True and validation_fraction > 0
    plt.title(f"{model_name} Training Loss Curve")
    plt.tight_layout()
    save_path = f"figures/plot_{model_name.lower()}_loss_curve.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def visualize_regression_quality(y_true, y_pred, model_name="MLP"):
    """
    Generates Parity Plot and Residual Plot.
    y_true/y_pred can be flattened (all outputs combined) for a global view.
    """
    # Flatten data if multi-output to see global performance
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    residuals = y_true_flat - y_pred_flat

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: Predicted vs Actual (Parity) ---
    sns.scatterplot(x=y_true_flat, y=y_pred_flat, ax=axes[0], alpha=0.5, color='blue')
    
    # Draw perfect diagonal line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    axes[0].set_title(f"{model_name}: Predicted vs Actual")
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- PLOT 2: Residuals vs Predicted ---
    sns.scatterplot(x=y_pred_flat, y=residuals, ax=axes[1], alpha=0.5, color='purple')
    
    # Draw zero line
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    
    axes[1].set_title(f"{model_name}: Residual Plot")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals (Actual - Predicted)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"figures/plot_{model_name.lower()}_quality.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    visualize_loocv_results()