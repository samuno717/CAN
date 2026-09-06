import time
import datetime
import torch
from loguru import logger
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, epochs, device):
    logger.info(f"Begin training loop ({epochs} epochs)...")
    global_start_time = time.perf_counter()

    for epoch in range(epochs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        model.train()
        total_loss = 0

        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for batch in batch_iterator:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_iterator.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)

        end_event.record()
        torch.cuda.synchronize()
        epoch_duration = start_event.elapsed_time(end_event) / 1000.0

        logger.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f} s. Average MSE: {avg_loss:.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()

    global_end_time = time.perf_counter()
    total_seconds = global_end_time - global_start_time
    formatted_time = str(datetime.timedelta(seconds=int(total_seconds)))

    logger.success(f"Training finished! Total time: {formatted_time} ({total_seconds:.2f} s)")

    return model