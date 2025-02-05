import subprocess
import os

if __name__ == "__main__":

    batch_sizes = [8, 16, 32]
    learning_rates = [1e-3, 1e-4, 1e-5]

    best_roc_auc = 0
    best_model_path = ""
    best_params = {}

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            output_dir = f"/kaggle/working/outputs/bs{batch_size}_lr{learning_rate}"
            
            # ✅ Run the training script
            cmd = [
                "python", "-m", "instageo.model.run",
                f"hydra.run.dir={output_dir}",
                "root_dir=/kaggle/input/geo-ai-hack",
                f"train.batch_size={batch_size}",
                f"train.learning_rate={learning_rate}",
                "train.num_epochs=5",
                "model.freeze_backbone=True",
                "mode=train",
                "train_filepath=train_split_reduced.csv",
                "valid_filepath=validation_split_reduced.csv"
            ]
            
            subprocess.run(cmd, capture_output=True, text=True)

            # ✅ Load the ROC-AUC score from the file
            roc_auc_path = os.path.join(output_dir, "roc_auc_score.txt")
            if os.path.exists(roc_auc_path):
                with open(roc_auc_path, "r") as f:
                    roc_auc = float(f.read().strip())

                # ✅ Check if this is the best model
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_path = os.path.join(output_dir, "best_model.pth")
                    best_params = {"batch_size": batch_size, "learning_rate": learning_rate}

    # ✅ Final Best Model and ROC-AUC
    print(f"Best ROC-AUC: {best_roc_auc}")
    print(f"Best Model Path: {best_model_path}")
    print(f"Best Hyperparameters: {best_params}")




# from instageo.model.run import main
# import sys
# from itertools import product

# # Define the parameter grid
# param_grid = {
#     "train.batch_size": [8, 16, 32],
#     "train.learning_rate": [1e-3, 1e-4, 1e-5],
# }

# # Generate all combinations of parameters
# keys, values = zip(*param_grid.items())
# param_combinations = [dict(zip(keys, v)) for v in product(*values)]

# # Run the gridsearch
# results = []
# for idx, params in enumerate(param_combinations):
#     print(f"Running configuration {idx + 1}/{len(param_combinations)}: {params}")
#     sys.argv = [
#         "run.py",
#         "--config-name=locust",
#         "hydra.run.dir=/kaggle/working/outputs/first_run",
#         "root_dir=/kaggle/input/",
#         f"train_filepath=train_split_reduced.csv",
#         f"valid_filepath=validation_split_reduced.csv",
#         "mode=train"
#     ] + [f"{key}={value}" for key, value in params.items()]

#     # Run and collect validation score
#     score = main()  # Assuming main() returns validation score
#     results.append({"params": params, "score": score})

# # Sort and print the best configuration
# best_result = max(results, key=lambda x: x["score"])
# print("Best configuration:", best_result)



# # """Optimize the Prithvi model by gridsearching over hyperparameters.
# # - learning rate
# # - batch size
# # - Adam optimizer's T0, beta1, beta2"""

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader

# # from instageo.model.model import PrithviSeg
# # from instageo.model.optim import Norm2D
# # from instageo.data import PrithviDataset
# # from instageo.data.transforms import get_transforms
# # from instageo.utils import get_device, get_logger



# # # logger = get_logger(__name__)


# # # def optimize_prithvi():
# # #     """Optimize the Prithvi model by gridsearching over hyperparameters."""
# # #     device = get_device()
# # #     logger.info(f"Using device: {device}")

# # #     # Initialize the dataset
# # #     dataset = PrithviDataset(
# # #         root_dir="data/prithvi",
# # #         split="train",
# # #         transforms=get_transforms(),
# # #     )
# # #     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # #     # Initialize the model
# # #     model = PrithviSeg()
# # #     model.to(device)

# # #     # Initialize the optimizer
# # #     optimizer = optim.AdamW(
# # #         model.parameters(),
# # #         lr=0.001,
# # #         betas=(0.9, 0.999),
# # #         eps=1e-8,
# # #         weight_decay=0.01,
# # #     )

# # #     # Initialize the loss function
# # #     criterion = nn.CrossEntropyLoss()

# # #     # Initialize the normalization module
# # #     norm = Norm2D(embed_dim=768)

# # #     # Train the model
# # #     for epoch in range(10):
# # #         model.train()
# # #         for i, (images, targets) in enumerate(dataloader):
# # #             images, targets = images.to(device), targets.to(device)

# # #             # Forward pass through the normalization module
# # #             images = norm(images)

# # #             # Forward pass through the model
# # #             outputs = model(images)

# # #             # Compute the loss
# # #             loss = criterion(outputs, targets)

# # #             # Backward pass and optimization
# # #             optimizer.zero_grad()
# # #             loss.backward()
# # #             optimizer.step()

# # #             if i % 10 == 0:
# # #                 logger.info(f"Epoch [{epoch}/{10}], Step [{i}/{len(dataloader)}], Loss: {loss.item()}")
# # #     logger.info("Optimization complete.")

# # # if __name__ == "__main__":
# # #     optimize_prithvi()
