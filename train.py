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
parser.add_argument('--gpu', type=bool, default=True, help='to use GPU')
parser.add_argument('--arch', type=str, default='vgg19', help='architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='initial hidden units')
parser.add_argument('--epochs', type=int, default=1, help='training epochs')

args = parser.parse_args()

data_dir = args.data_dir
node_hidden = args.hidden_units

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
arch=str(args.arch)     #'vgg19'   #ok    #'resnet152'   #ok     #densenet161' ok      #'alexnet' ok

if(arch=='vgg19'):
    model = models.vgg19(pretrained=True)
    input_size = model.classifier[0].in_features
    hidden_sizes = [node_hidden]
    output_size = 102

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Build classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier


if(arch=='resnet152'):
    model = models.resnet152(pretrained=True)
    input_size = 2048
    hidden_sizes = [node_hidden]
    output_size = 102

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.fc = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc

if(arch=='alexnet'):
    model = models.alexnet(pretrained=True)
    input_size = 9216
    hidden_sizes = [node_hidden]
    output_size = 102

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc


if(arch=='densenet161'):
    model = models.densenet161(pretrained=True)
    input_size = 2208
    hidden_sizes = [node_hidden]
    output_size = 102

    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier   #last layer of resnet called fc not classifier so removing model.classifier with model.fc


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
    for data in test_dataloaders:
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
              'input':input_size,
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
