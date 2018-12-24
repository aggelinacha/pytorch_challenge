"""!
@brief Training script for Image Classifier Project.

@author Aggelina Chatziagapi {aggelina.cha@gmail.com}
"""

import os
import json
import argparse
import numpy as np
import torch
from torchvision import datasets, models, transforms
from torch import nn as nn
from torch import optim as optim


def load_data(data_dir, batch_size=20, num_workers=0):
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
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=batch_size,
                                               num_workers=num_workers)
    return train_loader, valid_loader


def get_label_mapping(input_json="cat_to_name.json"):
    """!
    @brief Read json file with label mapping (from category label to
        category name).
    """
    with open(input_json, "r") as f:
        cat_to_name = json.load(f)

    n_classes = len(cat_to_name)

    return cat_to_name, n_classes


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


def build_model(n_classes, train_on_gpu=False):
    """!
    @brief Build network architecture using a pretrained VGG16 model.
    """
    vgg16 = models.vgg16(pretrained=True)

    # Freeze training for all "features" layers
    for param in vgg16.features.parameters():
        param.requires_grad = False

    n_inputs = vgg16.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, n_classes)
    vgg16.classifier[6] = last_layer

    # if GPU is available, move the model to GPU
    if train_on_gpu:
        vgg16.cuda()

    print(vgg16)

    return vgg16


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


def evaluate_model(model, valid_loader, criterion, n_classes,
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
    args_parser.add_argument('--num_workers', type=int,
                             default=0,
                             help="Number of subprocesses to use for "
                                  "data loading.")

    return args_parser.parse_args()


def main():
    """!
    @brief Main function for model training and evaluation.
    """
    args = parse_arguments()

    loader_tr, loader_val = load_data(args.input,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)
    train_on_gpu = check_cuda()
    _, num_classes = get_label_mapping()

    # Build model architecture
    model = build_model(num_classes, train_on_gpu=train_on_gpu)

    # Specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # Specify optimizer and learning rate
    optimizer = optim.SGD(model.classifier.parameters(),
                          lr=args.learning_rate)

    loss_val_min = np.Inf

    # Train model
    for epoch in range(1, args.n_epochs+1):

        loss_tr = train_model(model, loader_tr, criterion, optimizer,
                              train_on_gpu=train_on_gpu)
        loss_val, acc_val = evaluate_model(model, loader_val,
                                           criterion, num_classes,
                                           train_on_gpu=train_on_gpu)
        # Print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f}'
              '\tValidation Loss: {:.6f}'
              '\tValidation Accuracy: {:.2f}'
              .format(epoch,
                      loss_tr,
                      loss_val,
                      acc_val))

        # Save model if validation loss has decreased
        if loss_val <= loss_val_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). '
                  'Saving model ...'.format(loss_val_min,
                                            loss_val))
            torch.save(model.state_dict(), 'model.pt')
            loss_val_min = loss_val


if __name__ == '__main__':
    main()
