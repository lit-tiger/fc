import os
import sys
sys.path.append('E:\\feb\\feb\\test\\test\\test')
import numpy as np
import tensorflow as tf
import gzip
import os
import platform
import pickle
import cv2
from constant import constant


class dataset(object):
    def __init__(self, dataSetName, is_Non_IID, dtype=tf.float32):
        dype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype {}, expected uint8 or float32'.format(dtype))

        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_client = None
        self.train_label_client = None
        self.test_data = None
        self.test_label = None
        self.train_data_size = None
        self.test_data_size = None
        self.client_train_data_size = None
        self.is_Non_IID = is_Non_IID

        self._index_in_train_epoch = 0

        if self.name == "mnist" or "fmnist":
            self.mnist_dataset_construct(dtype)
            self.mnist_dataset_cut()
        if self.name == "cifar10":
            self.cifar10_dataset_construct(dtype)
            self.mnist_dataset_cut()

    def list_split(self, items, n):
        return [items[i:i+n] for i in range(0, len(items), n)]

    def mnist_dataset_cut(self):
        """
        labels = np.argmax(train_labels, axis=1)
        labels_upper = [0,0,0,0,0,0,0,0,0,0]
        for label in labels:
            labels_upper[label]=labels_upper[label]+1
        #for data_size_client_number in constant.CLIENT_CUT
        """
        


        if self.is_Non_IID == 0 :
            batch_size = constant.NODE_DATA_NUMBER
            self.train_data_client= []
            self.train_label_client = []
            start = 0
            end = 0
            for i in range (constant.CLIENT_NUMBER):
                end = start + batch_size
                if end > self.train_data_size:
                    order = np.arange(self.train_data_size)
                    np.random.shuffle(order)
                    self.train_data = self.train_data[order]
                    self.train_label = self.train_label[order]
                    start = 0
                    end = batch_size
                    assert batch_size <= self.train_data_size
                order = np.arange(batch_size)
                np.random.shuffle(order)
                self.train_data_client.append(self.train_data[order])
                self.train_label_client.append(self.train_label[order])
                start = end 
            return
        else :
            batch_size = int(constant.NODE_DATA_NUMBER/self.is_Non_IID)

        train_data_cut =[[] for _ in range(constant.CLASS_NUMBER)]
        train_label_cut =[[] for _ in range(constant.CLASS_NUMBER)]
        train_data_cut_array =[]
        train_label_cut_array =[]
        labels = np.argmax(self.train_label, axis=1)
        self.train_data_client  = []
        self.train_label_client = []

        for i in range (self.train_data_size):
            train_data_cut[labels[i]].append(self.train_data[i])
            train_label_cut[labels[i]].append(self.train_label[i])

        for i in range (constant.CLASS_NUMBER):
            train_data_cut_array.append(np.array(train_data_cut[i]))
            train_label_cut_array.append(np.array(train_label_cut[i]))
        

        """
        if self.is_Non_IID == 1 :
            for i in range (constant.CLIENT_NUMBER):
                self.train_data_client.append(train_data_cut[i%constant.CLASS_NUMBER])
                self.train_label_client.append(train_label_cut[i%constant.CLASS_NUMBER])                
        else :
            data = np.array([])
        label = np.array([])
        for j in range (self.is_Non_IID):
            if j == 0:
                data = train_data_cut_array[(i+j*(int(i/10)+1))%constant.CLASS_NUMBER]
                label = train_label_cut_array[(i+j*(int(i/10)+1))%constant.CLASS_NUMBER]
            else:
                data = np.concatenate((data,train_data_cut_array[(i+j*(int(i/10)+1))%constant.CLASS_NUMBER]), axis=0)              
                label = np.concatenate((label,train_label_cut_array[(i+j*(int(i/10)+1))%constant.CLASS_NUMBER]), axis=0)
            """

        if constant.CLIENT_NUMBER <=  constant.CLASS_NUMBER**(self.is_Non_IID+1): 
            flag = np.ones(constant.CLASS_NUMBER,dtype = np.int32)
            for i in range (constant.CLIENT_NUMBER):   
                    data = np.array([])
                    label = np.array([])
                    for j in range (self.is_Non_IID):
                        class_flag=(i+j*(int(i/10)+1))%constant.CLASS_NUMBER
                        start = flag[class_flag]
                        flag[class_flag] = flag[class_flag] + batch_size
                        if flag[class_flag] > train_data_cut_array[class_flag].shape[0]:
                            order = np.arange(train_data_cut_array[class_flag].shape[0])
                            np.random.shuffle(order)
                            train_data_cut_array[class_flag] = train_data_cut_array[class_flag][order]
                            train_label_cut_array[class_flag] = train_label_cut_array[class_flag][order]
                            start = 0
                            flag[class_flag] = batch_size
                            """
                            print("clint %d" %self.name)
                            print(batch_size)
                            print(self.train_data_size)
                            """
                            assert batch_size <= train_data_cut_array[class_flag].shape[0]
                        end = flag[class_flag]
                        if j == 0:
                            data = train_data_cut_array[class_flag][start: end]
                            label = train_label_cut_array[class_flag][start: end]
                        else:
                            data = np.concatenate((data,train_data_cut_array[class_flag][start: end]), axis=0)              
                            label = np.concatenate((label,train_label_cut_array[class_flag][start: end]), axis=0)
                    order = np.arange(batch_size*self.is_Non_IID)
                    np.random.shuffle(order)
                    self.train_data_client.append(data[order])
                    self.train_label_client.append(label[order])
                    #print(label[order])

    def mnist_dataset_construct(self, dtype):
        if self.name == "mnist":
            file_path = constant.MNIST_DATA_DIR
        else:
            file_path = constant.FMNIST_DATA_DIR
        
        train_images_path = os.path.join(file_path, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(file_path, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(file_path, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(file_path, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        """
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        """
        self.client_train_data_size = self.train_data_size/constant.CLIENT_NUMBER
        

        if dtype == tf.float32:
            train_images = train_images.astype(np.float32)
            train_images = np.multiply(train_images, 1.0 / 255.0)
            test_images = test_images.astype(np.float32)
            test_images = np.multiply(test_images, 1.0 / 255.0)

        order = np.arange(self.train_data_size)
        np.random.shuffle(order)
        self.train_data = train_images[order]
        self.train_label = train_labels[order]
        """
        if is_IID == 1:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            #print("train_labels = ")
            order = np.argsort(labels)
            #print(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        """
            
        self.test_data = test_images
        self.test_label = test_labels
        self.train_data = train_images[order]
        self.train_label = train_labels[order]
    
    def cifar10_dataset_construct(self, dtype):
        images, labels = [], []
        shape = (24, 24, 3)
        for filename in [constant.CIFAR10_DATA_DIR.format(i) for i in range(1, 6)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
            for i in range(len(cifar10[b'labels'])):
                old_image = np.reshape(cifar10[b'data'][i], (3, 32, 32))
                old_image = np.transpose(old_image, (1, 2, 0))
                old_image = old_image.astype(float)

                left = np.random.randint(old_image.shape[0] - shape[0] + 1)
                top = np.random.randint(old_image.shape[1] - shape[1] + 1)
                new_image = old_image[left: left + shape[0], top: top + shape[1], :]

                if np.random.random() < 0.5:
                    new_image = cv2.flip(new_image, 1)

                mean = np.mean(new_image)
                std = np.max([np.std(new_image),
                            1.0 / np.sqrt(new_image.shape[0] * new_image.shape[2] * new_image.shape[2])])
                new_image = (new_image - mean) / std

                images.append(new_image)
            labels += cifar10[b'labels']
            
        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int')
        order = np.arange(images.shape[0])
        np.random.shuffle(order)
        self.train_data = images[order]#.reshape(images.shape[0] , -1)
        self.train_label = dense_to_one_hot(labels[order]).astype(np.float32)

        images, labels = [], []
        with open(constant.CIFAR10_TEST_DIR, 'rb') as fo:
            if 'Windows' in platform.platform():
                cifar10 = pickle.load(fo, encoding='bytes')
            elif 'Linux' in platform.platform():
                cifar10 = pickle.load(fo, encoding='bytes')
        for i in range(len(cifar10[b'labels'])):
            image = np.reshape(cifar10[b'data'][i], (3, 32, 32))
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(float)
            images.append(image)
        labels += cifar10[b'labels']
        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int')
        self.test_label = dense_to_one_hot(labels).astype(np.float32)

        self.test_data = []
        shape = (24, 24, 3)
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            old_image = old_image[left: left + shape[0], top: top + shape[1], :]

            mean = np.mean(old_image)
            std = np.max([np.std(old_image),
                          1.0 / np.sqrt(images.shape[1] * images.shape[2] * images.shape[3])])
            new_image = (old_image - mean) / std

            self.test_data.append(new_image)

        self.test_data = np.array(self.test_data, dtype='float32')#.reshape(images.shape[0] , -1)
        self.train_data_size = self.train_data.shape[0]

    def dense_to_one_hot(labels_dense, num_classes=constant.CLASS_NUMBER):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels , dtype='float32') * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes), dtype= 'float32')
        labels_one_hot[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)   

if __name__ == '__main__':
    dataset = dataset("mnist", 2)
    print(dataset.train_data.shape)
    print(dataset.train_label.shape)
    print(dataset.test_data.shape)
    print(dataset.test_label.shape)
    print(type(dataset.train_data[0][0][0][0]))
    print(type(dataset.test_label[0][0]))
    """
    for i in range(constant.CLIENT_NUMBER):
        print(dataset.train_label_client[i].shape)
        order = np.argmax(dataset.train_label_client[i], axis=1)
        print(order[0: 10])
    """