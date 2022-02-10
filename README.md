# MFCN
We study and recommend a new technology to realize super-resolution reconstruction on a single image. Our model is trained iteratively on several different resolutions of the original training image, in which the resolution increases with the progress of training. We integrate the idea of momentum update in comparative learning into the design of Gan model structure. By constructing the main branch network and momentum update branch network model, we generate different levels of features in the same batch of training and compare the features.
# Acknowledgements
Our implementation is based on this implementation of the [SinGAN](https://github.com/tamarott/SinGAN) paper.
