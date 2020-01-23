import numpy as np
import matplotlib.pyplot as plt

def dataloader(_type):  # _type: train or test
    data = []
    filename = 'hw3_{}.txt'.format(_type)
    f = open(filename)
    for line in f:
        line = line.replace('\n','').split(' ')
        line = line[1:]
        data.append(line)
    f.close()
    data = np.array(data).astype(np.float)
    label_array = data[:,-1].reshape(-1, 1).astype('int')
    data_array = data[:, :-1] 
    bias = np.ones((data_array.shape[0], 1))
    data_array = np.concatenate((bias, data_array), axis=1)
    return data_array, label_array

def sigmoid(x):
    theta = 1 / (1+np.exp(-1*x))
    return theta

def Gradient_descent(w, training_data, training_data_label, testing_data, testing_data_label, lr, T): 
    weight_matrix = w
    Ein_list, Eout_list, T_list = [], [], []

    for t in range(T):
        T_list.append(t)
        wx = np.matmul(training_data, weight_matrix)  # (1000, 1)
        wxy = wx * training_data_label # (1000, 1)
        xy = training_data * training_data_label  # (1000, 21)
        delta_Ein = (1 / training_data.shape[0]) * np.sum( sigmoid(-wxy) * (-1) * xy, axis=0)
        delta_Ein = delta_Ein.reshape((-1,1))
        weight_matrix = weight_matrix - lr * delta_Ein
        
        ein = Error_rate(weight_matrix, training_data, training_data_label)
        eout = Error_rate(weight_matrix, testing_data, testing_data_label)
        Ein_list.append(ein)
        Eout_list.append(eout)

    return Ein_list, Eout_list, T_list

def Stochastic_gradient_descent(w, training_data, training_data_label, testing_data, testing_data_label, lr, T): # T=2000
    weight_matrix = w
    n = 0
    Ein_list, Eout_list, T_list = [], [], []

    for t in range(T):
        T_list.append(t)
        if t >= training_data.shape[0]:
            n = 0
        wx = np.matmul(training_data[n, :].reshape((1, -1)), weight_matrix) # (1, 1)
        wxy = wx * training_data_label[n, :].reshape((1, 1)) # (1, 1)
        xy = training_data[n, :].reshape((1, -1)) * training_data_label[n, :].reshape((1, 1)) # (1, 21)
        delta_Ein = sigmoid(-wxy) * (-1) * xy 
        delta_Ein = delta_Ein.reshape((-1, 1))
        weight_matrix = weight_matrix - lr * delta_Ein
        n += 1

        ein = Error_rate(weight_matrix, training_data, training_data_label)
        eout = Error_rate(weight_matrix, testing_data, testing_data_label)
        Ein_list.append(ein)
        Eout_list.append(eout) 

    return Ein_list, Eout_list, T_list

def Error_rate(weight_matrix, data, data_label):
    N = data_label.shape[0]
    pred = np.dot(data, weight_matrix) 
    pred[pred >= 0] = 1
    pred[pred < 0] = -1

    err_cnt = np.sum( pred != data_label.reshape((N,1)) )
    return err_cnt / N 

if __name__ == "__main__":
    training_data, training_data_label = dataloader('train') # training_data: (1000,21)   training_data_label: (1000, 1)
    testing_data, testing_data_label = dataloader('test') # testing_data: (3000, 21)  testing_data_label: (3000, 1) 
    
    training_data_num, training_data_length = training_data.shape[0], training_data.shape[1]
    weight_matrix = np.zeros((training_data_length, 1)) # (21, 1)
    T, lr_gd, lr_sgd = 2000, 0.01, 0.001
    Ein_list_gd, Eout_list_gd, Ein_list_sgd, Eout_list_sgd, T_list = [], [], [], [], []
    
    Ein_list_gd, Eout_list_gd, T_list = Gradient_descent(weight_matrix, training_data, training_data_label, testing_data, testing_data_label, lr_gd, T)

    weight_matrix = np.zeros((training_data_length, 1)) # (21, 1)
    Ein_list_sgd, Eout_list_sgd, T_list = Stochastic_gradient_descent(weight_matrix, training_data, training_data_label, testing_data, testing_data_label, lr_sgd, T)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(T_list, Eout_list_gd, 'r', label='curve for Eout(GD)')
    plt.plot(T_list, Eout_list_sgd, 'b', label='curve for Eout(SGD)')
    
    plt.legend(loc='best')
    plt.title("Eout")
    plt.xlabel("T")
    plt.ylabel("Error rate")
    
    plt.subplot(212)
    plt.plot(T_list, Eout_list_gd, 'r', label='curve for Ein(GD)')
    plt.plot(T_list, Ein_list_sgd, 'b', label='curve for Ein(SGD)')

    plt.legend(loc='best')
    plt.title("Ein")
    plt.xlabel("T")
    plt.ylabel("Error rate")
    
    plt.show()


   