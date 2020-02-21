# sporeseg

Segmentation of spores

## Installing the environment:

Download this repository and unzip it. This provides the necessary scritps and notebooks to run the code. To install the package itself, create an environment, activate it and pip install the package:

```
conda create -n spores pip
source activate spores
pip install git+https://github.com/guiwitz/sporeseg.git@master#egg=sporeseg

```
To update the package after new releases, no need to download the repository, just type:
```
source activate spores
pip install --upgrade git+https://github.com/guiwitz/sporeseg.git@master#egg=sporeseg
```

For a developement installation, cd to the downloaded repository and type
```
pip install -e .
```

The -e flag is important if you want to be able to edit the code and be able to reload the package without re-installing it.




