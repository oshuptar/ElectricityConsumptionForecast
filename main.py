from src.utils import (get_device,
                        get_batch_size,
                        get_hidden_layer1_size,
                        get_hidden_layer2_size,
                        get_learning_rate,
                        build_comparison_dataframe)
from src.dataset import (get_dataset, get_data_loaders, get_dataframe)
from src.model import ForecastModel
from src.train import (train_model, mae_criterion, mape_criterion, rmse_criterion, evaluate_model)
from src.plotting import (plot_week_15min, plot_year_hourly)
import torch.nn as nn
import torch.optim as optim

def main():
    device = get_device()
    window_size = 10
    train_dataset, test_dataset = get_dataset(window_size=window_size)
    train_loader, test_loader = get_data_loaders(train_dataset=train_dataset, test_dataset=test_dataset)
    criterion = nn.MSELoss()
    print("Q5:")
    model = ForecastModel(in_features=window_size)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(f"Model architecture: {model}")
    train_model(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion,
                eval_criteria=[("MAE", mae_criterion), ("RMSE", rmse_criterion), ("MAPE", mape_criterion)],
                optimizer=optimizer, device=device, epochs=5)

    print("\nQ7: - running the hyperparameter tunning")
    best_result = greedy_experiment(criterion, device)

    print("\nQ8 - plotting predicted vs actual on a test dataset")
    best_model = best_result["model"]
    results_df = build_comparison_dataframe(
        model=best_model,
        test_loader=test_loader,
        test_df=get_dataframe('electricity_test.csv'),
        window_size=window_size,
        device=device
    )
    plot_week_15min(results_df)
    plot_year_hourly(results_df)

def greedy_experiment(criterion, device):
    window_size = 10
    train_dataset, test_dataset = get_dataset(window_size=window_size)
    history = []
    for hd1_size in get_hidden_layer1_size():
        for hd2_size in get_hidden_layer2_size():
            for lr in get_learning_rate():
                for batch_size in get_batch_size():
                   model = ForecastModel(in_features=window_size, hd1=hd1_size, hd2=hd2_size)
                   print("\n" + ("-" * 50))
                   print(f"\nhidden1={hd1_size}, hidden2={hd2_size}, lr={lr}, batch_size={batch_size}")
                   train_loader, test_loader = get_data_loaders(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size)
                   optimizer = optim.SGD(model.parameters(), lr=lr)
                   trained_model = train_model(model=model, train_loader=train_loader,test_loader=test_loader, criterion=criterion,eval_criteria=[("MAE", mae_criterion)],optimizer=optimizer, device=device, epochs=3)
                   mae = evaluate_model(trained_model, test_loader, mae_criterion, device)
                   history.append({
                        "hidden1_size": hd1_size,
                        "hidden2_size": hd2_size,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "mae": mae,
                        "model": trained_model
                    })
    history.sort(key=lambda x: x["mae"])

    if len(history) >= 3:
        print("\nBest configurations:")
        for i in range(3):
            print(f"{i + 1}. {history[i]}")
    else:
        print("\nBest configuration:")
        print(history[0])

    return history[0]


    
if __name__ == "__main__":
    main()