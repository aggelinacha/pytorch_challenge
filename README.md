### Short description

This is a fork from udacity's repo: https://github.com/udacity/pytorch_challenge.git, including the PyTorch Scholarship Challenge final project.

The goal of this project is the training of an image classifier to recognize different species of flowers, based on the 102 Category Flower Dataset (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).


### Download data

```wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip```

Extract the above zip under `pytorch_challenge/` directory.


### Install requirements

```pip3 install -r requirements.txt```


### Train model

```python3 train.py -i flower_data -e 30 -o model.pt --gpu```


### Make predictions

```python3 predict.py -i flower_data/valid/1/image_06749.jpg -ckpt model.pt --gpu --plot```

