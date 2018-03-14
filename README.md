# Functional Map of the World Challenge Entry Code

This is the 1st-place entry for the [Functional Map of the World challenge](https://www.iarpa.gov/challenges/fmow.html) organized by [IARPA](https://www.iarpa.gov/).


## Contest overview

The goal of this contest was to create a geospatial classifier to determine the nature of an object visible in a series of satellite images given its bounding box. There are 62 object classes with diverse sizes, including visually similar classes such as fire station and police station.


## Solution overview

The solution is based on an ensemble of 12 deep convolutional network classifiers fine-tuned from a generic pre-trained [Dual-Path Network](https://github.com/cypw/DPNs). They use a variety of hyperparameters, scaling methods and augmentation methods. Some of them include a validation set while others do not.

For example, 2 of the 12 models simply resize the object to be classified to a fixed pixel size regardless of its physical size. The other models use an overall more effective power-law scaling which provides more context for small objects but more detail for large objects. The simple models, while being less effective on their own, are included because they bring diversity to the ensemble, which usually helps increase accuracy.

These models use the [PyTorch](http://pytorch.org/) framework. Additionally, the challenge organizers commissioned [JHU/APL](http://www.jhuapl.edu/) to build a baseline model and provided model weights already trained on the challenge dataset. Including this completely independent model in the final submission helped provide greater diversity and had the further benefit of not being counted toward the training time limit.


## License

This code is placed under the Apache License 2.0.

It also includes slightly modified versions of the [fMoW baseline](https://github.com/fmow/baseline) and [PyTorch Pretrained DPN](https://github.com/rwightman/pytorch-dpn-pretrained) repositories. Both are licensed under Apache 2.0.


## Running the code

### Requirements

- nvidia-docker 2.0 or higher must be installed.
- The fMoW-rgb dataset must be downloaded and extracted into folder `data`, including the `val_false_detection` supplemental data.
- You will need a few hundred gigabytes of temporary disk space in a location such as `/tmp/fmow`.
- You must clone this repository into a directory such as `fmow_pfr`.


### Building and running the Docker image

Build and start the Docker image with:

```
docker build -t fmow_pfr fmow_pfr
nvidia-docker run -v data:/data:ro -v /tmp/fmow:/wdata --ipc=host -it fmow_pfr
```

Inside the Docker container shell, you can directly execute the trained models on a test set:
```
./test.sh /data/train /data/test name_of_output_file
```
Note that in accordance with the challenge specification, a `.txt` suffix will be appended to the name you choose.

You can also run training by calling
```
./train.sh /data/train
```
The trained model weights are stored in `trained_models/*.pth`. Calling `test.sh` again after training will use the newly created models.
