import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


class LogisticRegression:
    def __init__(self):
        self.train_set_x_orig, self.train_set_y, self.test_set_x_orig, self.test_set_y, self.classes=self.load_dataset()


    def load_dataset(self):
        train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    
        test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    
        classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


    def show_img(self):
        index = 25
        plt.imshow(self.train_set_x_orig[index])
        # plt.show()
        # 使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
        # print("train_set_y=",self.train_set_y[:,index])
        print("train_set_y=", np.squeeze(self.train_set_y[:, index]))
        print("y=" + str(self.train_set_y[:, index]) + ", it's a " + self.classes[np.squeeze(self.train_set_y[:, index])].decode(
            "utf-8") + " picture")

    def model(self):
        # train_set_x_orig 是一个维度为(m_​​train,num_px,num_px,3）的数。
        m_train = self.train_set_y.shape[1]  # 训练集里图片的数量。
        m_test = self.test_set_y.shape[1]  # 测试集里图片的数量。
        num_px = self.train_set_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度（均为64x64）。

        # 现在看一看我们加载的东西的具体情况
        print("训练集的数量: m_train = " + str(m_train))
        print("测试集的数量 : m_test = " + str(m_test))
        print("每张图片的宽/高 : num_px = " + str(num_px))
        print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print("训练集_图片的维数 : " + str(self.train_set_x_orig.shape))
        print("训练集_标签的维数 : " + str(self.train_set_y.shape))
        print("测试集_图片的维数: " + str(self.test_set_x_orig.shape))
        print("测试集_标签的维数: " + str(self.test_set_y.shape))

        # 为了方便，我们要把维度为（64，64，3）的numpy数组重新构造为（64 x 64 x 3，1）的数组
        # X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
        # 将训练集的维度降低并转置。
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        # 将测试集的维度降低并转置。
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

if __name__=='__main__':
    lr = LogisticRegression()
    # lr.show_img()
    lr.model()
