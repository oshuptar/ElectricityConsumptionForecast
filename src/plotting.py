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

def plot_year_hourly(results_df):
    # alternatively, we can take sum aggregate to see the total consumption per month

    #set_index - sets index to timestamp column and resamples "actual" and "predicted" columns
    year_df = (
        results_df.set_index("timestamp")[["actual", "predicted"]]
        .resample("1h")
        .mean()
        .reset_index()
    )

    #   Alternative syntax:
    #     temp_df = results_df.set_index("timestamp")
    #     temp_df = temp_df[["actual", "predicted"]]
    #     temp_df = temp_df.resample("1h").mean()
    #     hourly_df = temp_df.reset_index()

    plt.figure(figsize=(14,5))
    plt.plot(year_df["timestamp"], year_df["actual"], label = "Actual")
    plt.plot(year_df["timestamp"], year_df["predicted"], label="Predicted")
    plt.title("Actual vs Predicted - Single Year (1-hour resolution)")
    plt.xlabel("Timestamp")
    plt.ylabel("Consumption")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "year_plot.png")
    plt.savefig(output_path)
    print(f"The plot file was saved to {output_path}")