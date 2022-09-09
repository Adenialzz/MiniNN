import gzip
import os
import pickle

# =================================== datasets ================================= #

class Dataset:

    def __init__(self, data_dir, **kwargs):
        self._train_set = None
        self._valid_set = None
        self._test_set = None

        self._save_paths = [os.path.join(data_dir, url.split("/")[-1])
                            for url in self._urls]

        self._download()
        self._parse(**kwargs)  # lgtm [py/init-calls-subclass]

    def _download(self):
        for url, checksum, save_path in zip(
                self._urls, self._checksums, self._save_paths):
            download_url(url, save_path, checksum)

    def _parse(self, **kwargs):
        raise NotImplementedError

    @property
    def train_set(self):
        return self._train_set

    @property
    def valid_set(self):
        return self._valid_set

    @property
    def test_set(self):
        return self._test_set

    @staticmethod
    def one_hot(targets, n_classes):
        return np.eye(n_classes, dtype=np.float32)[np.array(targets).reshape(-1)]


class MNIST(Dataset):

    def __init__(self, data_dir, one_hot=True):
        self._urls = ("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz",)
        self._checksums = ("98100ca27dc0e07ddd9f822cf9d244db",)
        self._n_classes = 10
        super().__init__(data_dir, one_hot=one_hot)

    def _parse(self, **kwargs):
        save_path = self._save_paths[0]
        with gzip.open(save_path, "rb") as f:
            train, valid, test = pickle.load(f, encoding="latin1")

        if kwargs["one_hot"]:
            train = (train[0], self.one_hot(train[1], self._n_classes))
            valid = (valid[0], self.one_hot(valid[1], self._n_classes))
            test = (test[0], self.one_hot(test[1], self._n_classes))

        self._train_set, self._valid_set, self._test_set = train, valid, test



# ========================== iterator ===========================  #

from collections import namedtuple
import numpy as np

Batch = namedtuple("Batch", ["inputs", "targets"])

class BaseIterator:

    def __call__(self, inputs, targets):
        raise NotImplementedError


class BatchIterator(BaseIterator):

    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        indices = np.arange(len(inputs))
        if self.shuffle:
            np.random.shuffle(indices)

        starts = np.arange(0, len(inputs), self.batch_size)
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[indices[start: end]]
            batch_targets = targets[indices[start: end]]
            yield Batch(inputs=batch_inputs, targets=batch_targets)

# ===================== for downloading datasets ========================== #
import hashlib
import os
from urllib.error import URLError
from urllib.request import urlretrieve

def show_progress(blk_num, blk_sz, tot_sz):
    percentage = 100.0 * blk_num * blk_sz / tot_sz
    print(f"Progress: {percentage:.1f} %", end="\r", flush=True)


def md5_checksum(file_path):
    with open(file_path, "rb") as fileobj:
        checksum = hashlib.md5(fileobj.read()).hexdigest()
    return checksum


def download_url(url, file_path, checksum):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if os.path.exists(file_path):
        if md5_checksum(file_path) == checksum:
            print(f"{file_path} already exists.")
            return
        print("Wrong checksum!")

    try:
        print(f"Downloading {url} to {file_path}")
        urlretrieve(url, file_path, show_progress)
    except URLError:
        raise RuntimeError("Error downloading resource!")
    except KeyboardInterrupt:
        print("Interrupted")
