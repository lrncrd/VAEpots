# VAEpots
A Convolutional Variational Autoencoder for features extraction of archaeological pottery profiles.
The dataset consists of about 5000 vessels from central Tyrrhenian Italy.

Supplementary material for the paper *A deep Variational Convolutional Autoencoder for unsupervised features extraction of ceramic profiles.  A case study from central Italy*

#### Site distribution 

[<img src="https://github.com/lrncrd/VAEpots/blob/main/presentation/maps.jpg" width="700"/>](image.png)

Distribution of selected sites used in the analysis

#### Dataset example 

[<img src="https://github.com/lrncrd/VAEpots/blob/main/presentation/batch.jpg" width="500"/>](image.png)

#### Train and Test Loss

[<img src="https://github.com/lrncrd/VAEpots/blob/main/presentation/loss_img.jpg" width="800"/>](image.png)

#### PCA and UMAP embeddings

[<img src="https://github.com/lrncrd/VAEpots/blob/main/presentation/Reduction.jpg" width="800"/>](image.png)

## Installation

[Anaconda](https://www.anaconda.com/) is recommended.

you can use the .yml file (VAEpots.yml) to create an Anaconda enviroment. Open Anaconda Prompt:

```bash
conda env create -f VAEpots.yml
```

for further information https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


the Pytorch library is not included and must be installed according to the requirements of your system (https://pytorch.org/)
