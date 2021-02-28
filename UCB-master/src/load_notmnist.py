import pickle
import tensorflow
file_name = "../data/NotMNIST/train_test_small_notmnist"
data = pickle.load(open(file_name,"rb"))
data_test = data['image_test']
shape = data_test.shape
data_reshape = data_test.reshape(shape[0],1,28,28)
x =data_reshape[0][0]