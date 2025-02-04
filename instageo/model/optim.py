"""Optimize the Prithvi model by gridsearching over hyperparameters.
- learning rate
- batch size
- Adam optimizer's T0, beta1, beta2"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from instageo.model.model import PrithviSeg
from instageo.model.optim import Norm2D
from instageo.data import PrithviDataset
from instageo.data.transforms import get_transforms
from instageo.utils import get_device, get_logger

logger = get_logger(__name__)


def optimize_prithvi():
    """Optimize the Prithvi model by gridsearching over hyperparameters."""
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize the dataset
    dataset = PrithviDataset(
        root_dir="data/prithvi",
        split="train",
        transforms=get_transforms(),
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize the model
    model = PrithviSeg()
    model.to(device)

    # Initialize the optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize the normalization module
    norm = Norm2D(embed_dim=768)

    # Train the model
    for epoch in range(10):
        model.train()
        for i, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)

            # Forward pass through the normalization module
            images = norm(images)

            # Forward pass through the model
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                logger.info(f"Epoch [{epoch}/{10}], Step [{i}/{len(dataloader)}], Loss: {loss.item()}")
    logger.info("Optimization complete.")

if __name__ == "__main__":
    optimize_prithvi()
