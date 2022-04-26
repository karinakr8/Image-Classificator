import torch


def train_single_epoch(model, dataloader, loss_fn, optimizer, device, scheduler):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    for i, data in enumerate(dataloader):

        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        print(outputs.shape)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        _, prediction = torch.max(outputs, 1)

        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    acc = correct_prediction / total_prediction
    print(f'Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')


def train(model, data_loader, loss_fn, optimiser, device, epochs, scheduler):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, scheduler)
        print("---------------------------")
    print("Finished training")
