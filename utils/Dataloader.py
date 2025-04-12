import numpy as np
import pickle

class CIFAR10Dataloader:
    def __init__(self, path_dir="cifar-10-batches-py", n_valid=5000, batch_size=32):
        # 加载训练集和测试集
        self.x_train, self.y_train = self._load_cifar10_train(path_dir)
        self.x_test, self.y_test = self._load_cifar10_test(path_dir)

        # 分割验证集
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.train_valid_split(
            self.x_train, self.y_train, n_valid
        )
        self.batch_size = batch_size

    @staticmethod
    def _load_cifar10_batch(file):
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            # 数据格式转换：通道在前转为HWC并压平
            images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            images = images.reshape(-1, 32 * 32 * 3).astype(np.float32) / 255.0
            labels = np.eye(10)[data[b'labels']]  # One-hot编码
            return images, labels

    def _load_cifar10_train(self, path_dir):
        images, labels = [], []
        for i in range(1, 6):
            batch_path = f"{path_dir}/data_batch_{i}"
            img, lbl = self._load_cifar10_batch(batch_path)
            images.append(img)
            labels.append(lbl)
        return np.concatenate(images), np.concatenate(labels)

    def _load_cifar10_test(self, path_dir):
        return self._load_cifar10_batch(f"{path_dir}/test_batch")

    @staticmethod
    def train_valid_split(x, y, n_valid):
        indices = np.random.permutation(x.shape[0])
        return (
            x[indices[n_valid:]],
            y[indices[n_valid:]],
            x[indices[:n_valid]],
            y[indices[:n_valid]],
        )

    def generate_train_batch(self):
        n_samples = self.x_train.shape[0]
        indices = np.random.permutation(self.x_train.shape[0])
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def generate_valid_batch(self):
        n_samples = self.x_valid.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_valid[batch_indices], self.y_valid[batch_indices]

    def generate_test_batch(self):
        n_samples = self.x_test.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_test[batch_indices], self.y_test[batch_indices]
    # 保持原有的generate_*_batch方法
