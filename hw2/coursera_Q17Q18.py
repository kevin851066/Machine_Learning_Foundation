import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    x[x>=0] = 1
    x[x<0] = -1
    return x

def GetTheta(x):
    cp_x = x.copy()
    cp_x.sort()
    theta = np.zeros(len(cp_x))
    for i in range(len(cp_x)-1):
        theta[i] = ( cp_x[i+1] + cp_x[i] ) / 2
    theta[len(cp_x)-1] = 1

    return theta

def GenerateTrainingData(n):
    x_data, y_data = np.zeros(n), np.zeros(n)
    x_data = np.random.uniform(-1, 1, n)
    noise = sign( np.random.uniform(-0.2, 0.8, n) )
    y_data = sign( x_data.copy() ) * noise

    return x_data, y_data

def experiment(exp_time, data_size):
    arr_ein, arr_eout = 0, 0
    result_arr = []
    for t in range(exp_time):
        s, best_theta = 0, 0
        x, y, theta = np.zeros(data_size), np.zeros(data_size), np.zeros(data_size)
        x, y = GenerateTrainingData(data_size)
        theta = GetTheta(x)
        e_in = np.zeros((2, data_size))     # s = 1 or s = -1
        for i in range(data_size):
            h = sign( x - theta[i] )
            e_in[0][i] = np.sum(h != y) / data_size     # s = 1
            e_in[1][i] = np.sum(-h != y) / data_size    # s = -1

        min0, min1 = np.min(e_in[0]), np.min(e_in[1])
        if min0 < min1:
            s = 1
            best_theta = theta[np.argmin(e_in[0])]
        else:
            s = -1
            best_theta = theta[np.argmin(e_in[1])]

        e_in = np.min(e_in)
        e_out = 0.5 + 0.3*s*(abs(best_theta) - 1)
        arr_ein += e_in
        arr_eout += e_out

        result = e_in - e_out
        result_arr.append( round(result, 2) )

    arr_ein = arr_ein / exp_time
    arr_eout = arr_eout / exp_time
    print("For question 17, Average Ein: ", arr_ein)
    print("For question 18, Average Eout: ", arr_eout)

if __name__ == '__main__':
    experiment(5000, 20)


