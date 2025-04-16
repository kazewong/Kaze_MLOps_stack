import tarfile
import requests
import pickle
import grain.python as grain
from jaxtyping import Int, Array
import jax.numpy as jnp

class CIFAR10DataSource(grain.RandomAccessDataSource):

    def read_cifar(self, path: str) -> tuple[Int[Array, "50000 3 32 32"], Int[Array, "50000 10"]]:
        inputs = []
        labels = []
        for i in range(1, 6):
            with open(path+'data_batch_'+str(i), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                inputs.append(jnp.array(data[b'data']))
                labels.append(jnp.array(data[b'labels']))
        
        return jnp.concatenate(inputs).reshape(-1, 3, 32, 32), jnp.concatenate(labels)

    def __init__(self, data_dir: str = 'data/'):
        super().__init__()
        try:
            data = self.read_cifar(data_dir + 'cifar-10-batches-py/')
            self.images = data[0]
            self.labels = data[1]
        except FileNotFoundError:
            print('Downloading CIFAR-10 dataset...')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            r = requests.get(url)
            with open(data_dir+'cifar-10-python.tar.gz', 'wb') as f:
                f.write(r.content)
            with tarfile.open(data_dir+'cifar-10-python.tar.gz') as tar:
                tar.extractall(data_dir)
            data = self.read_cifar(data_dir + 'cifar-10-batches-py/')
            self.images = data[0]
            self.labels = data[1]

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
        }
    
    def __len__(self):
        return len(self.images)