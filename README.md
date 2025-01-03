# GrEASE

<p align="center">

[//]: # (    <img src="https://github.com/shaham-lab/SpectralNet/blob/main/figures/twomoons.png">)

This is the official PyTorch implementation of GrEASE from the paper ["Generalizable Spectral Embedding with Applications to UMAP]().<br>

[//]: # (See out [GitHub repository]&#40;https://github.com/shaham-lab/GrEASE&#41; for more information.)

The main application of GrEASE is NUMAP, a generalizable version of UMAP. The code for NUMAP can be found [here](https://github.com/shaham-lab/NUMAP).

## Installation
To install the package, simply use the following command:

```bash
pip install grease-embeddings
```

## Usage

The basic functionality is quite intuitive and easy to use, e.g.,

```python
from grease import GrEASE

grease = GrEASE(n_components=10)  # n_components is the number of dimensions in the low-dimensional representation
grease.fit(X)  # X is the dataset and it should be a torch.Tensor
X_reduced = grease.transfrom(X)  # Get the low-dimensional representation of the dataset
Y_reduced = grease.transform(Y)  # Get the low-dimensional representation of a test dataset

```

You can read the code docs for more information and functionalities.<br>

Out of many applications, GrEASE can be used for generalizable Fiedler vector and value approximation, and Diffusion Maps approximation. The following is examples of how to use GrEASE for these applications:

### Fiedler vector and value approximation

```python
from grease import GrEASE

grease = GrEASE(n_components=1)
fiedlerVector = grease.fit_transform(X)
fiedlerValue = grease.get_eigenvalues()
```

### Diffusion Maps approximation

```python
from grease import GrEASE

grease = GrEASE(n_components=10)
diffusionMaps = grease.fit_transform(X, t=5)  # t is the diffusion time
```

## Running examples

In order to run the model on the moon dataset, you can either run the file, or using the command-line command:<br>
`python -m examples.reduce_moon`<br>
This will run the model on the moon dataset and plot the results.

The same can be done for the circles dataset:<br>
`python -m examples.reduce_circles`<br>




[//]: # (## Citation)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (@inproceedings{shaham2018,)

[//]: # (author = {Uri Shaham and Kelly Stanton and Henri Li and Boaz Nadler and Ronen Basri and Yuval Kluger},)

[//]: # (title = {SpectralNet: Spectral Clustering Using Deep Neural Networks},)

[//]: # (booktitle = {Proc. ICLR 2018},)

[//]: # (year = {2018})

[//]: # (})

[//]: # ()
[//]: # (```)
