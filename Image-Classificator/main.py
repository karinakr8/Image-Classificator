# Pirmasis GMM laboratorinis darbas

# Atliko:    Karina Krapaitė, 5 gr.
# LSP:       1911065
# Variantas: 6
import os

import torch
import training
import inference

from torch import nn
from model import NeuralNet
from torch.utils.data import DataLoader, random_split
from data import AudioDataset

# Train Dataset ----------------------------------------------------------------------
# Used for learning (by the model)
# Valid Dataset ----------------------------------------------------------------------
# Used to provide an unbiased evaluation of a model fitted on the training dataset while tuning model hyper parameters
# Test Dataset -----------------------------------------------------------------------
# Used to provide an unbiased evaluation of a final model fitted on the training dataset

BATCH_SIZE = 16
EPOCHS = 9
LEARNING_RATE = 0.001
NUM_WORKERS = 2

ANNOTATIONS_FILE = "C:/Users/karin/datasets/speech_commands_v0.01/list.txt"
AUDIO_DIR = "C:/Users/karin/datasets/speech_commands_v0.01"

TEST_ANNOTATIONS_FILE = "C:/Users/karin/datasets/test_data/list.txt"
#TEST_DIR = "C:/Users/karin/datasets/test_data"

SAMPLE_RATE = 16000
NUM_SAMPLES = 65000
DURATION = 1000
SHIFT_LIMIT = 0.1
CHANNEL = 2

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}.")

dataset = AudioDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, DURATION,
                       SHIFT_LIMIT, CHANNEL)
#test_dataset = AudioDataset(TEST_ANNOTATIONS_FILE, TEST_DIR, SAMPLE_RATE, 3, DURATION,
                           # SHIFT_LIMIT, CHANNEL)

# Random split of 80:20 between training and validation

num_train = round(NUM_SAMPLES * 0.8)
num_val = NUM_SAMPLES - num_train
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

print(f'Training dataset:   {len(train_dataset)}')
print(f'Validating dataset: {len(val_dataset)}')
# print(f'Testing dataset:    {len(test_dataset)}')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=3, shuffle=False)


def create_model():
    model = NeuralNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=EPOCHS,
                                                    anneal_strategy='linear')

    training.train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS, scheduler)

    torch.save(model.state_dict(), "model.pth")
    print("Trained feed forward net saved at model.pth")


def test_model(dataloader):
    model = NeuralNet().to(device)
    state_dict = torch.load("model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html

    print("I N T E R E N C E -----------------------------------")
    inference.inference(model, dataloader, device)


def main():
    print(f"There are {len(dataset)} samples in the dataset.")
    x, y = dataset[1]
    print(y)
    classes = dataset.unique()
    print(f'Classes({len(classes)}): {classes}')

    create_model()
    # test_model(val_dataloader)
    # test_model(test_dataloader)


if __name__ == '__main__':
    main()
