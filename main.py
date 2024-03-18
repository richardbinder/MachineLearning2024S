import openml
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import pandas as pd


if __name__ == "__main__":
    # Get dataset by name
    dataset = openml.datasets.get_dataset('Stock-Information', download_data=True, download_qualities=True, download_features_meta_data=True)

    # Get the data itself as a dataframe (or otherwise)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(X)

    print("test")
