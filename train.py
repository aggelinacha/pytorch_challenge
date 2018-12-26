"""!
@brief Training script for Image Classifier Project.

@author Aggelina Chatziagapi {aggelina.cha@gmail.com}
"""

import os
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
from torchvision import datasets, models, transforms
from torch import nn as nn
from torch import optim as optim


def load_data(data_dir, batch_size=20, n_workers=0):
    """
    @brief Load train and validation data and apply transformations.

    @param batch_size (\a int) How many samples per batch to load.
    @param num_workers (\a int) Number of subprocesses to use for data
        loading.

    @returns \b train_loader DataLoader for train data.
    @returns \b valid_loader DataLoader for validation data.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,
                                      transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir,
                                    transform=val_transforms)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=n_workers)
    valid_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=batch_size,
                                               num_workers=n_workers)
    return train_loader, valid_loader


def get_label_mapping(input_json="cat_to_name.json"):
    """!
    @brief Read json file with label mapping (from category label to
        category name).
    """
    with open(input_json, "r") as f:
        cat_to_name = json.load(f)

    return cat_to_name


def check_cuda():
    """!
    @brief Check if CUDA is available.
    """
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    return train_on_gpu


def build_model_vgg(n_classes, pretrained="vgg16", hidden_1=4096,
                    hidden_2=4096, drop_prob=0.5, train_on_gpu=False):
    """!
    @brief Build network architecture using a pretrained VGG model.
    """
    if pretrained == "vgg16":
        model = models.vgg16(pretrained=True)
    elif pretrained == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        raise NotImplementedError("Use of {} as pretrained network for "
                                  "feature extraction is not "
                                  "implemented yet".format(pretrained))

    # Freeze training for all features layers
    for param in model.features.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
        ('0', nn.Linear(n_inputs, hidden_1)),
        ('1', nn.ReLU(inplace=True)),
        ('2', nn.Dropout(drop_prob)),
        ('3', nn.Linear(hidden_1, hidden_2)),
        ('4', nn.ReLU(inplace=True)),
        ('5', nn.Dropout(drop_prob)),
        ('6', nn.Linear(hidden_2, n_classes))]))

    model.classifier = classifier

    # if GPU is available, move the model to GPU
    if train_on_gpu:
        model.cuda()

    print(model)

    return model


def train_model(model, train_loader, criterion, optimizer,
                train_on_gpu=False):
    """!
    @brief Train model.
    """
    train_loss = 0.0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass
        # compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass
        # compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.dataset)

    return train_loss


def evaluate_model(model, valid_loader, criterion, n_classes=10,
                   train_on_gpu=False):
    """!
    @brief Validate model.
    """
    valid_loss = 0.0
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    model.eval()

    for batch_idx, (data, target) in enumerate(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass
        # compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        if not train_on_gpu:
            correct = np.squeeze(correct_tensor.numpy())
        else:
            correct = np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(target.size(0)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    valid_loss = valid_loss / len(valid_loader.dataset)
    valid_acc = 100. * np.sum(class_correct) / np.sum(class_total)

    return valid_loss, valid_acc


def parse_arguments():
    """!
    @brief Parse Arguments for training a CNN with multiple datasets
           and corresponding tasks.
    """
    args_parser = argparse.ArgumentParser(description="Image classifier"
                                                      " training")
    args_parser.add_argument('-i', '--input', type=str,
                             required=True,
                             help="Path of the flower dataset.")
    args_parser.add_argument('-e', '--n_epochs', type=int,
                             default=10,
                             help="Number of epochs to train the "
                                  "model.")
    args_parser.add_argument('-lr', '--learning_rate', type=float,
                             default=0.001,
                             help="Learning rate.")
    args_parser.add_argument('-b', '--batch_size', type=int,
                             default=20,
                             help="Batch size.")
    args_parser.add_argument('--n_workers', type=int,
                             default=0,
                             help="Number of subprocesses to use for "
                                  "data loading.")
    args_parser.add_argument('--pretrained', type=str,
                             default="vgg16",
                             help="Model type to use as a pretrained "
                                  "network for feature extraction.")
    args_parser.add_argument('--hidden_1', type=int,
                             default=4096,
                             help="Number of hidden units for the "
                                  "classifier's 1st linear layer.")
    args_parser.add_argument('--hidden_2', type=int,
                             default=4096,
                             help="Number of hidden units for the "
                                  "classifier's 2nd linear layer.")
    args_parser.add_argument('--dropout', type=float,
                             default=0.5,
                             help="Dropout probability.")
    args_parser.add_argument('-o', '--output', type=str,
                             default='model.pt',
                             help="Path to save the model checkpoint.")

    return args_parser.parse_args()


def main():
    """!
    @brief Main function for model training and evaluation.
    """
    args = parse_arguments()

    loader_tr, loader_val = load_data(args.input,
                                      batch_size=args.batch_size,
                                      n_workers=args.n_workers)
    train_on_gpu = check_cuda()
    cat_to_name = get_label_mapping()
    num_classes = len(cat_to_name)

    # Build model architecture
    model = build_model_vgg(num_classes,
                            pretrained=args.pretrained,
                            hidden_1=args.hidden_1,
                            hidden_2=args.hidden_2,
                            drop_prob=args.dropout,
                            train_on_gpu=train_on_gpu)

    # Specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # Specify optimizer and learning rate
    optimizer = optim.SGD(model.classifier.parameters(),
                          lr=args.learning_rate)

    loss_val_min = np.Inf

    # Train model
    for epoch in range(args.n_epochs):

        loss_tr = train_model(model,
                              loader_tr,
                              criterion,
                              optimizer,
                              train_on_gpu=train_on_gpu)
        loss_val, acc_val = evaluate_model(model,
                                           loader_val,
                                           criterion,
                                           n_classes=num_classes,
                                           train_on_gpu=train_on_gpu)
        # Print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} '
              '\tValidation Loss: {:.6f}'
              '\tValidation Accuracy: {:.2f}'
              .format(epoch + 1,
                      loss_tr,
                      loss_val,
                      acc_val))

        # Save model if validation loss has decreased
        if loss_val <= loss_val_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). '
                  'Saving model ...'.format(loss_val_min,
                                            loss_val))
            torch.save(model.state_dict(), args.output)
            loss_val_min = loss_val


if __name__ == '__main__':
    main()
