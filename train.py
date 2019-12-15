#python train.py --data_dir dir_path --gpu True --learning_rate 0.001

# Imports here
import numpy as np
import torch
import argparse
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# arguments to be passed
parser = argparse.ArgumentParser(description='Flower Classification')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='saved checkpoint')
parser.add_argument('--gpu', type=bool, default=False, help='to use GPU')
parser.add_argument('--arch', type=str, default='VGG', help='architecture')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='initial hidden units')
parser.add_argument('--epochs', type=int, default=3, help='training epochs')

args = parser.parse_args()

data_dir = args.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

nThreads = 4
batch_size = 8
use_gpu = torch.cuda.is_available()

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
image_datasets = dict()
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

image_datasets['train'] = train_datasets
image_datasets['valid'] = valid_datasets
image_datasets['test'] = test_datasets


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = dict()
train_dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size)
test_dataloaders = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)

dataloaders['train'] = train_dataloaders
dataloaders['valid'] = valid_dataloaders
dataloaders['test']  = test_dataloaders

# TODO: Build the model
model = models.vgg19(pretrained=True)
num_features = model.classifier[0].in_features

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Build classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_features, 1024)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(1024, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc3', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# Set negative log loss as the criteria
criterion = nn.NLLLoss()

# Only training the classifier parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train the network
epochs = args.epochs
batch = 32
steps = 0
train_loss = 0

device = torch.device('cuda:0' if args.gpu else 'cpu')
model.to(device)

for e in range(epochs):
    model.train()

    for inputs, labels in iter(train_dataloaders):

        inputs = Variable(inputs)
        targets = Variable(labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model.forward(inputs)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        steps += 1
        if steps % batch== 0:
            model.eval()
            accuracy = 0
            test_loss = 0

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(valid_dataloaders):

                    inputs = Variable(inputs)
                    labels = Variable(labels)

                    inputs, labels = inputs.to(device), labels.to(device)

                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).item()

                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss/batch),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloaders)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloaders)))

            train_loss = 0

            model.train()

# TODO: Do validation on the test set
valid = 0
total = 0

model.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        valid += (predicted == labels).sum().item()

print('Accuracy of the network: {}' .format(valid / total))

# TODO: Save the checkpoint
model.class_to_idx = train_datasets.class_to_idx

check_point = {'arch':args.arch,
              'input':num_features,
              'output':102,
              'epochs':args.epochs,
              'learning_rate':args.learning_rate,
              'dropout':0.3,
              'batch_size':32,
              'classifier':classifier,
              'state_dict':model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(check_point, args.save_dir)
