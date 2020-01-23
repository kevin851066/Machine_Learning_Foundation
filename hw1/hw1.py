import time
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def sign(x):
    result = -1 if x <= 0. else 1
    return result

def pla(eta, dataset): # for traditional pla
    w = np.zeros(5)
    idx, correct_num, update_num = 0, 0, 0
    while True:
        x = dataset[idx]
        label = x[5]
        if correct_num == len(dataset):
            break
        if sign( w.T.dot(x[:5]) ) != label:
            w = w + eta * label * x[:5]
            update_num += 1
            correct_num = 0
        else:
            correct_num += 1
        if idx >= len(dataset) - 1:
            idx = 0
        else:
            idx += 1
    return update_num

def check(w, dataset): # for the testing of pla with pocket algorithm
    err_num = 0
    for x in dataset:
        label = x[5]
        if sign( w.T.dot(x[:5]) ) != label:
            err_num += 1  
    err_rate = err_num / float(len(dataset))
    return err_rate

def modified_non_greedy_pla(train_data, test_data, update_num): # for pla with pocket(non-greedy) algorithm
    w = np.zeros(5)
    idx, err_rate = 0, 0

    un = 0
    while True:
        x = train_data[idx]
        label = x[5]
        if sign( w.T.dot(x[:5]) ) != label:
            w = w + label * x[:5]
            un += 1
            err_rate = check(w, test_data)
        if idx >= len(train_data) - 1:
            idx = 0
        else:
            idx += 1
        if un == update_num:
            break
    return err_rate

def modified_greedy_pla(train_data, test_data, update_num): # for pla with pocket(greedy) algorithm
    w_best, w = np.zeros(5), np.zeros(5)
    least_err_rate, idx = 0, 0

    err_rate = check(w, test_data)
    w_best = w
    least_err_rate = err_rate

    un = 0
    while True:
        x = train_data[idx]
        label = x[5]
        if sign( w.T.dot(x[:5]) ) != label:
            w = w + label * x[:5]
            un += 1
            err_rate = check(w, test_data)
            if err_rate < least_err_rate:
                least_err_rate = err_rate
                w_best = w
        if idx >= len(train_data) - 1:
            idx = 0
        else:
            idx += 1
        if un == update_num:
            break
    return least_err_rate

def dataloader(_type, _num):  # type: train or test, num: 6 or 7
    data = []
    filename = '{}{}.txt'.format(_type, _num)
    f = open(filename)
    for line in f:
        line = line.replace('\t',' ').replace('\n','').split(' ')
        line.insert(0, '1')
        line = [float(i) for i in line]
        data.append(line)
    f.close()
    data = np.array(data) 
    return data

data = dataloader("train", "6")
update_num_list = []
for i in range(1126):
    np.random.shuffle(data)
    update_num = pla(1, data)
    update_num_list.append(update_num)

avg_update_num = sum(update_num_list) / float(len(update_num_list))
print("Problem 6", "average update number: ", avg_update_num)

# c1 = Counter(update_num_list)
# print(c1)

# plt.figure(1)
# plt.subplot(211)
# plt.bar(c1.keys(), c1.values())   

# plt.title("Q6")

training_data = dataloader("train", "7")
testing_data = dataloader("test", "7")
err_rate_list = []
update_num = 100
for i in range(1126):
    np.random.shuffle(training_data)
    err_rate = modified_greedy_pla(training_data, testing_data, update_num)
    err_rate_list.append( round(err_rate, 2) )

avg_err_rate = sum(err_rate_list) / float(len(err_rate_list))
print("Problem 7, avg_err_rate: ", avg_err_rate)

# _max = max(err_rate_list)
# _min = min(err_rate_list)
# bins = (_max - _min) / 0.01 + 1
# print(_max,_min)

# c2 = Counter(err_rate_list)
# print(c2)

# plt.subplot(212)

# plt.title("Q7")
# plt.hist(err_rate_list, bins=int(bins), align='left', range=[_min, _max], rwidth=0.5)   

# plt.show()

err_rate_list = []
for i in range(1126):
    np.random.shuffle(training_data)
    err_rate = modified_non_greedy_pla(training_data, testing_data, update_num)
    err_rate_list.append( round(err_rate, 2) )

avg_err_rate = sum(err_rate_list) / float(len(err_rate_list))
print("Problem 8, avg_err_rate: ", avg_err_rate)



