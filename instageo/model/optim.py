from instageo.model.run import main
import sys
from itertools import product

# Define the parameter grid
param_grid = {
    "train.batch_size": [8, 16, 32],
    "train.learning_rate": [1e-3, 1e-4, 1e-5],
}

# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

# Run the gridsearch
results = []
for idx, params in enumerate(param_combinations):
    print(f"Running configuration {idx + 1}/{len(param_combinations)}: {params}")
    sys.argv = [
        "run.py",
        "--config-name=locust",
        "hydra.run.dir=/kaggle/working/outputs/first_run",
        "root_dir=/kaggle/input/",
        f"train_filepath=train_split_reduced.csv",
        f"valid_filepath=validation_split_reduced.csv",
        "mode=train"
    ] + [f"{key}={value}" for key, value in params.items()]

    # Run and collect validation score
    score = main()  # Assuming main() returns validation score
    results.append({"params": params, "score": score})

# Sort and print the best configuration
best_result = max(results, key=lambda x: x["score"])
print("Best configuration:", best_result)



# """Optimize the Prithvi model by gridsearching over hyperparameters.
# - learning rate
# - batch size
# - Adam optimizer's T0, beta1, beta2"""

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# from instageo.model.model import PrithviSeg
# from instageo.model.optim import Norm2D
# from instageo.data import PrithviDataset
# from instageo.data.transforms import get_transforms
# from instageo.utils import get_device, get_logger



# # logger = get_logger(__name__)


# # def optimize_prithvi():
# #     """Optimize the Prithvi model by gridsearching over hyperparameters."""
# #     device = get_device()
# #     logger.info(f"Using device: {device}")

# #     # Initialize the dataset
# #     dataset = PrithviDataset(
# #         root_dir="data/prithvi",
# #         split="train",
# #         transforms=get_transforms(),
# #     )
# #     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# #     # Initialize the model
# #     model = PrithviSeg()
# #     model.to(device)

# #     # Initialize the optimizer
# #     optimizer = optim.AdamW(
# #         model.parameters(),
# #         lr=0.001,
# #         betas=(0.9, 0.999),
# #         eps=1e-8,
# #         weight_decay=0.01,
# #     )

# #     # Initialize the loss function
# #     criterion = nn.CrossEntropyLoss()

# #     # Initialize the normalization module
# #     norm = Norm2D(embed_dim=768)

# #     # Train the model
# #     for epoch in range(10):
# #         model.train()
# #         for i, (images, targets) in enumerate(dataloader):
# #             images, targets = images.to(device), targets.to(device)

# #             # Forward pass through the normalization module
# #             images = norm(images)

# #             # Forward pass through the model
# #             outputs = model(images)

# #             # Compute the loss
# #             loss = criterion(outputs, targets)

# #             # Backward pass and optimization
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()

# #             if i % 10 == 0:
# #                 logger.info(f"Epoch [{epoch}/{10}], Step [{i}/{len(dataloader)}], Loss: {loss.item()}")
# #     logger.info("Optimization complete.")

# # if __name__ == "__main__":
# #     optimize_prithvi()
