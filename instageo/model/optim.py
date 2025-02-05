import subprocess
import os
from sklearn.model_selection import ParameterGrid

def gridsearch(model, trainer, train_loader, valid_loader):
    param_grid = {
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'batch_size': [8, 16, 32]
    }

    grid = ParameterGrid(param_grid)
    results = []
    output_results_dir = "/home/poscalice/GeoAI/InstaGeo-E2E-Geospatial-ML/instageo/data/optim/"
    # Ensure the output directory exists
    os.makedirs(output_results_dir, exist_ok=True)
    results_path = os.path.join(output_results_dir, "gridsearch.txt")
    with open(results_path, "w") as f:
        f.write(f"Grid search results for {model.__class__.__name__}\n")

    for params in grid:
        model.learning_rate = params['learning_rate']
        model.batch_size = params['batch_size']
        trainer.fit(model, train_loader, valid_loader)
        results.append((params, trainer.callback_metrics['val_loss']))
        # create and write in the txt file
        with open(results_path, "a") as f:
            f.write(f"{results[-1]}\n")

    best_params = sorted(results, key=lambda x: x[1])[0]
    # append to the txt file
    with open(results_path, "a") as f:
        f.write(f'Best parameters: {best_params[0]}, Validation Loss: {best_params[1]}\n')

    print(f'Best parameters: {best_params[0]}, Validation Loss: {best_params[1]}')
