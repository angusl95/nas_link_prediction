# Neural Architecture Search for Knowledge Base Link Prediction

First install KBC and preprocess the datasets as they suggest:

## Installation
Create a conda environment with pytorch cython and scikit-learn :
```
conda create --name kbc_env python=3.7
source activate kbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the kbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are download, add them to the package data folder by running :
```
python kbc/process_datasets.py
```

This will create the files required to compute the filtered metrics.

If you are encountering errors, check that the datasets.py files are pointing at the correct paths for the data.

## Architecture Search and Evaluation
There is one folder provded for each of the non-strided and strided search spaces. In each case, the script `train_search.py` performs architecture search, and `train.py` trains existing architectures, which can be specified in `genotypes.py`.

Notes:
The command line argument `--interleaved` toggles the interleaving of input embeddings
Learning rate decay is enabled by default, but I did not use it for the results in the my thesis: to ensure settings match the thesis, ensure that `--learning_rate` and `--learning_rate_min` are set to the same value.
The argument `--layers` controls the number of repeated cells in the architecture (we use 1 throughout my thesis), and `--steps` controls the number of learned operations within each cell (we typically use 5 in my thesis).
By default, the argument `--reg` controls the amount of N3 regularisation used. In my thesis this is set to 0, and we instead use `--weight_decay` for regularisation.

## License
kbc is CC-BY-NC licensed, as found in the LICENSE file.
