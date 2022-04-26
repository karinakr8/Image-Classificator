import torch


def inference(model, dataloader, device):
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for data in dataloader:

            inputs = data[0].to(device)
            labels = data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)

            _, prediction = torch.max(outputs, 1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            print(f'Predicted: {prediction}')
            print(f'Expected: {labels}')

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')