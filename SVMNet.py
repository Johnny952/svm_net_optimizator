import numpy as np
import random
from sklearn import svm

kernels = ['rbf', 'sigmoid', 'poly']
degrees = [1, 2, 3, 4]
# gamma_range = np.logspace(-9, 3, 13)

def extract_classes(Data):
    """
    Extract classes from data labels.
    :param Data: matrix of examples.
    :return: list of classes.
    """
    try:
        classes = np.unique(Data[:, -1])
    except IndexError:
        classes = np.unique(Data)
    except TypeError:
        classes = np.unique(Data)
    return classes


class SVMNet(object):
    def __init__(self):
        self.left = None
        self.left_classes = np.array([])
        self.right = None
        self.right_classes = np.array([])
        self.SVM = None
        self.C = 0
        self.kernel = ''
        self.degree = 1
        # self.gamma = 1

    def new_net(self, classes):
        """
        Create a SVM net randomly.
        :param classes: list of classes.
        :return: the object SVMNet (self).
        """
        # Spit classes in two lists, left and right classes
        number_of_classes = len(classes)
        rand = random.randint(0, number_of_classes - 2) + 1
        for i in range(rand):
            aux = random.choice(classes)
            classes = np.setdiff1d(classes, aux)
            self.left_classes = np.concatenate([self.left_classes, aux], axis=None)
        self.right_classes = np.copy(classes)
        # Create a net in left and right if their number of classes is more than one
        if len(self.left_classes) > 1:
            self.left = SVMNet().new_net(self.left_classes)
        if len(self.right_classes) > 1:
            self.right = SVMNet().new_net(self.right_classes)
        # Set parameters of the net
        self.kernel = random.choice(kernels)
        self.C = random.random() * 1000  # C in range [0,1000]
        self.degree = random.choice(degrees)
        # self.gamma = random.choice(gamma_range)
        self.SVM = svm.SVC(kernel=self.kernel, degree=self.degree, C=self.C, gamma='auto')
        return self

    def print(self):
        """
        Prints the parameters of the net.
        """
        print("Classes", [self.left_classes, self.right_classes], "\t", )
        print("C = ", self.C, "\t", )
        print("Kernel = ", self.kernel, "\t", )
        print("Degree = ", self.degree, "\t")
        # print "Gamma = ", self.gamma
        if self.left is not None:
            self.left.print()
        if self.right is not None:
            self.right.print()

    def fit(self, data, labels_orig):
        """
        Trains the net.
        :param data: matrix of examples without labels.
        :param labels_orig: list of labels for each example in data.
        """
        labels = np.copy(labels_orig)
        left_data = []
        left_labels = []
        right_data = []
        right_labels = []
        # For each example in data, if it's label is in the left node, place a 0 in labels list, otherwise a 1.
        # And separate this examples in two groups depending on the labels.
        for i in range(len(labels)):
            if labels[i] in self.left_classes:
                left_data.append(data[i, :])
                left_labels = np.concatenate([left_labels, labels[i]], axis=None)
                labels[i] = 0
            elif labels[i] in self.right_classes:
                right_data.append(data[i, :])
                right_labels = np.concatenate([right_labels, labels[i]], axis=None)
                labels[i] = 1
            else:
                raise Exception("Class not found in database: ", labels[i])
        self.SVM.fit(data, labels)
        if self.left is not None:
            self.left.fit(np.array(left_data), left_labels)
        if self.right is not None:
            self.right.fit(np.array(right_data), right_labels)

    def classify(self, data):
        """
        Classifies the data.
        :param data: matrix if examples unlabeled.
        :return: list of predicted labels.
        """
        if data.size != 0:
            predicted_labels = self.SVM.predict(data)
            left_data = []
            right_data = []
            for p in range(data.shape[0]):
                if predicted_labels[p] == 0:
                    left_data.append(data[p])
                elif predicted_labels[p] == 1:
                    right_data.append(data[p])
                else:
                    raise Exception("Label value not allowed: ", predicted_labels[p])
            left_data = np.array(left_data)
            right_data = np.array(right_data)
            if self.left is not None:
                left_labels = self.left.classify(left_data)
            else:
                left_labels = np.array([self.left_classes[0]] * left_data.shape[0])
            if self.right is not None:
                right_labels = self.right.classify(right_data)
            else:
                right_labels = np.array([self.right_classes[0]] * right_data.shape[0])
            i = 0
            j = 0
            for k in range(len(predicted_labels)):
                if predicted_labels[k] == 0:
                    predicted_labels[k] = left_labels[i]
                    i += 1
                else:
                    predicted_labels[k] = right_labels[j]
                    j += 1
            return predicted_labels