# coding=utf-8
import sys

import torch
import torch.onnx as torch_onnx
import torchvision
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CNN

# Ensure result is reproducible
RANDOM_SEED = 42

# Define hyper parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# Step 1: Set up target metrics for evaluating training

# Define a target loss metric to aim for
target_accuracy = 0.9

# instantiate classifier and scaler

# Step 2: Perform training for model
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Training data
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valtset = datasets.MNIST('./MNIST_data/', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valtset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate the classifier
cnn = CNN().cuda()

# Get classes
classes = valloader.dataset.class_to_idx

# Define loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE, eps=1e-08)

# Write training metrics to Tensorboard
writer = SummaryWriter('metrics/tensorboard/', comment='CNN')

images, labels = next(iter(trainloader))
grid = torchvision.utils.make_grid(images.cuda())

writer.add_graph(model=cnn, input_to_model=images.cuda())
writer.add_image('MNIST', grid, global_step=0)
writer.close()

# train model
epochs = 5

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    cnn.train()
    train_correct = 0
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        inputs, target = batch
        output = cnn(inputs.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        y_pred = output.argmax(dim=1, keepdim=True)
        train_correct += y_pred.eq(target.cuda().view_as(y_pred)).sum().item()

    train_loss /= len(trainloader)
    train_accuracy = 100 * train_correct / len(trainloader.dataset)

    cnn.eval()
    val_correct = 0
    for batch in valloader:
        inputs, target = batch
        output = cnn(inputs.cuda())
        loss = criterion(output, target.cuda())
        val_loss += loss.data.item()
        y_pred = output.argmax(dim=1, keepdim=True)
        val_correct += y_pred.eq(target.cuda().view_as(y_pred)).sum().item()

    val_loss /= len(valloader)
    val_accuracy = 100 * val_correct / len(valloader.dataset)

    print(f"Epoch {epoch + 1} :: Train/Loss {round(train_loss, 3)} :: Train/Accuracy {round(train_accuracy, 3)}")
    print(f"Epoch {epoch + 1} :: Val/Loss {round(val_loss, 3)} :: Val/Accuracy {round(val_accuracy, 3)}")

    writer.add_scalars('Loss', {'Train/Loss': train_loss,
                                'Test/Loss': val_loss}, epoch + 1)

    writer.add_scalars('Accuracy', {'Train/Accuracy': train_accuracy,
                                    'Val/Accuracy': val_accuracy}, epoch + 1)
    writer.flush()

writer.close()

# Step 3: Evaluate the quality o.gropuf the trained model
# Only persist the model if we have passed our desired threshold
if val_accuracy < target_accuracy:
    sys.exit('Training failed to meet threshold')

# Step 4: Persist the trained model in ONNX format in the local file system along with any significant metrics

# persist model
rand_images, _ = next(iter(trainloader))
torch_onnx.export(cnn,
                  rand_images.cuda(),
                  'model.onnx',
                  verbose=False)
