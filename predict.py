"""!
@brief Predict image class using a trained model.

@author Aggelina Chatziagapi {aggelina.cha@gmail.com}
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from train import load_checkpoint
from utils import check_cuda, get_label_mapping


def process_image(image_path):
    """!
    @brief Scale, crop, and normalize a PIL image for a PyTorch model.
        Returns a Pytorch tensor.

    @param image_path (\a str) Path of the input image to load.

    @returns \b image (\a PyTorch tensor) Pre-processed image.
    """
    size_scale = 256
    size_crop = 224

    # Load PIL image
    pil_image = Image.open(image_path)

    # Resize
    height, width = pil_image.size
    pil_image = pil_image.resize(
        (max(size_scale, int(size_scale * height / width)),
         max(size_scale, int(size_scale * width / height))))

    # Crop
    center_h = pil_image.size[0] // 2
    center_v = pil_image.size[1] // 2
    left = center_h - (size_crop / 2)
    right = center_h + (size_crop / 2)
    upper = center_v - (size_crop / 2)
    lower = center_v + (size_crop / 2)
    pil_image = pil_image.crop(box=(left, upper, right, lower))
    np_image = np.array(pil_image, dtype=float)
    np_image = np_image / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Convert to PyTorch tensor
    return torch.Tensor(np_image.transpose(2, 0, 1))


def imshow(image, ax=None, title=None):
    """!
    @brief Imshow for Pytorch tensor.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    if title:
        ax.set_title(title)

    return ax


def plot_predictions(probs, labels, topk=5, ax=None):
    """!
    @brief Plot probabilities (bar graph) for top-k predicted classes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ypos = np.arange(topk)
    ax.barh(ypos, probs)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)


def predict(image, model, topk=5):
    """!
    @brief Predict the class (or classes) of an image using a trained
        deep learning model.

    @param image (\a PyTorch tensor) Batch of images.
    @param model (\a Pytorch model) Trained model.

    @returns \b prob_k Probabilities of the top K classes.
    @returns \b ind_k Indeces of the top K classes.
    """
    model.eval()

    # Get model output
    output = model(image)
    prob = torch.nn.Softmax(dim=1)(output).data
    prob_k, ind_k = prob.topk(topk)

    prob_k = prob_k.cpu().numpy().squeeze()
    ind_k = ind_k.cpu().numpy().squeeze()

    return prob_k, ind_k


def map_classes(ind, mapping):
    """!
    @brief Map indices to classes.
    """
    classes = [mapping[idx] for idx in ind]
    return classes


def print_results(classes, probs):
    """!
    @brief Print top-k predicted classes along with their probabilities.
    """
    print("Predicted class: '{}' \t(p = {})".format(classes[0],
                                                    probs[0]))
    print("")
    print("Top K classes:")
    print("")
    for i, (c, p) in enumerate(zip(classes, probs), 1):
        print("{}: '{}' \t(p = {})".format(i, c, p))


def parse_arguments():
    """!
    @brief Parse Arguments for predicting the image class(es) using a
        trained image classifier.
    """
    args_parser = argparse.ArgumentParser(description="Predict")
    args_parser.add_argument('-i', '--image_path', type=str,
                             required=True,
                             help="Path of an image.")
    args_parser.add_argument('-ckpt', '--checkpoint', type=str,
                             required=True,
                             help="Path of the model checkpoint.")
    args_parser.add_argument('-k', '--top_k', type=int,
                             default=5,
                             help="Top K classes to predict.")
    args_parser.add_argument('--plot', action='store_true',
                             help="Imshow image along with the top K "
                                  "predicted classes.")
    args_parser.add_argument('--cat_to_name', type=str,
                             default="cat_to_name.json",
                             help="Json file that maps the class values"
                                  " to category names.")
    args_parser.add_argument('--gpu', action='store_true',
                             help="Test on GPU.")

    return args_parser.parse_args()


def main():
    """!
    @brief Main function for predicting the image class(es) using a
        trained model.
    """
    args = parse_arguments()
    test_on_gpu = (args.gpu and check_cuda())
    cat_to_name = get_label_mapping(input_json=args.cat_to_name)

    # Load checkpoint
    model, ckpt_dict = load_checkpoint(args.checkpoint,
                                       train_on_gpu=test_on_gpu)
    idx_to_class = {idx: cat_to_name[c]
                    for c, idx in ckpt_dict['class_to_idx'].items()}

    # Pre-process image
    image = process_image(args.image_path)
    image = torch.unsqueeze(image, 0)
    if test_on_gpu:
        image = image.cuda()

    # Get actual label
    label = os.path.basename(os.path.dirname(args.image_path))
    label = cat_to_name[label]

    # Predictions - top K classes
    prob_k, ind_k = predict(image, model, topk=args.top_k)
    classes_k = map_classes(ind_k, idx_to_class)
    print("True label: '{}'".format(label))
    print("")
    print_results(classes_k, prob_k)

    # Plot image and predictions
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
        imshow(torch.squeeze(image.cpu()),
               ax=ax1,
               title=label)
        plot_predictions(prob_k,
                         classes_k,
                         ax=ax2,
                         topk=args.top_k)
        plt.show()


if __name__ == '__main__':
    main()
