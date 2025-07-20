import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, epochs=50, device='cuda'):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Loss computation
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

        # Optional: evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"           Validation Loss: {val_loss:.4f}")

@torch.no_grad()
def evaluate(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0

    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = criterion(predictions, targets)
        val_loss += loss.item()

    return val_loss / len(val_loader)
