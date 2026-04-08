import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_week_15min(results_df, start = None):
    if start is None:
        start = results_df["timestamp"].min()

    end = start + pd.Timedelta(days=7)
    week_df = results_df[(results_df["timestamp"] >= start) & (results_df["timestamp"] < end)]
    plt.figure(figsize=(14, 5))
    plt.plot(week_df["timestamp"], week_df["actual"], label="Actual")
    plt.plot(week_df["timestamp"], week_df["predicted"], label="Predicted")
    plt.title("Actual vs Predicted - Single Week (15-minute resolution)")
    plt.xlabel("Timestamp")
    plt.ylabel("Consumption")
    plt.legend()
    plt.xticks(rotation=45) # rotates the labels on the x-axis by 45 degrees
    plt.tight_layout()

    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "week_plot.png")
    plt.savefig(output_path)
    print(f"The plot file was saved to {output_path}")
