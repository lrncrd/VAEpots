# VAEpots
A Convolutional Variational Autoencoder for features extraction of archaeological pottery profiles

The dataset includes about 5000 ceramic profiles from central tyrrhenian Italy.

Supplementary material for the paper *A deep Variational Autoencoder for unsupervised features extraction of ceramic profiles. A case study from central Italy*

[![DOI](https://img.shields.io/badge/doi-10.1016%2Fj.jas.2022.105640%20-green)](https://doi.org/10.1016/j.jas.2022.105640)

### Citation
```Latex
@article{cardarelli_deep_2022,
	title = {A deep variational convolutional Autoencoder for unsupervised features extraction of ceramic profiles. A case study from central Italy},
	volume = {144},
	issn = {03054403},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S030544032200098X},
	doi = {10.1016/j.jas.2022.105640},
	pages = {105640},
	journaltitle = {Journal of Archaeological Science},
	shortjournal = {Journal of Archaeological Science},
	author = {Cardarelli, Lorenzo},
	urldate = {2022-07-10},
	date = {2022-08},
	langid = {english},
}
```





### Site distribution
Distribution of some of the sites used in the analysis. Complete map in high-resolution [here](https://github.com/lrncrd/VAEpots/blob/main/site_distribution.png). 

[<img src="https://github.com/lrncrd/VAEpots/blob/main/presentation/maps.jpg" width="500"/>](image.png)

### Sample dataset
A *batch* of profiles edited from archaeological drawings

[<img src="https://github.com/lrncrd/VAEpots/blob/main/presentation/batch.jpg" width="300"/>](image.png)

### Reconstruction examples
Reconstruced profiles from Test Set

[<img src="https://github.com/lrncrd/VAEpots/blob/main/imgs/Rec_20.jpg" width="700"/>](image.png)

### Multivariate analysis
Multivariate analysis on Latent Dimension

[<img src="https://github.com/lrncrd/VAEpots/blob/main/presentation/Reduction.jpg" width="800"/>](image.png)

## Installation

[Anaconda](https://www.anaconda.com/) is recommended.

you can use the .yml file (VAEpots.yml) to create an Anaconda enviroment. Open Anaconda Prompt:

```bash
conda env create -f VAEpots.yml
```

for further information https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


the Pytorch library is not included and must be installed according to the requirements of your system (https://pytorch.org/)

## Interactive scatterplot visualization

At the moment, you need to download the file (https://github.com/lrncrd/VAEpots/blob/main/Interactive_scatterplot.html) and open it locally. I am working on a better solution
