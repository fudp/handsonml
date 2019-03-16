import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from zlib import crc32
from sklearn.base import BaseEstimator, TransformerMixin


# Hardcoded column numbers for specific fields in the dataset
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


def fetch_housing_data(housing_url, housing_path):
    """Fetch the housing data and uncompress them"""
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio, random_state=42):
    """Basic train-test splitter."""
    n_samples = len(data)
    shuffled_indices = np.random.permutatiln(n_samples)
    test_size = int(test_ratio * n_samples)
    idx_test = shuffled_indices[:test_size]
    idx_train = shuffled_indices[test_size:]
    return data.iloc[idx_train], data.iloc[idx_test]


def test_set_check(identifier, test_ratio):
    # Check https://docs.python.org/2/library/zlib.html#zlib.crc32
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        rooms_per_population = X[:, rooms_ix] / X[:, population_ix]
        bedrooms_per_population = X[:, bedrooms_ix] / X[: population_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, bedrooms_per_room,
                         population_per_household, rooms_per_population,
                         bedrooms_per_population]
        else:
            return np.c_[X, rooms_per_household, population_per_household,
                         rooms_per_population, bedrooms_per_population]
