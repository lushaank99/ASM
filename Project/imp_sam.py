import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

e = 2.71828

def get_rand_number(min_value, max_value):
    range = max_value - min_value
    choice = random.uniform(0,1)
    return min_value + range*choice

def f_of_x(x):
    return (e**(-1))/(1+(x-1)**3)

def g_of_x(x, A, lamda):
    return A * math.pow(e, -1 * lamda * x)

def inverse_G_of_r(r, lamda):
    return (-1 * math.log(float(r))) / lamda

def get_IS_variance(lamda, num_samples):
    A = lamda
    int_max = 5

    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += (f_of_x(x) / g_of_x(x, A, lamda)) ** 2

    sum_of_sqs = running_total / num_samples

    # get squared average
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x) / g_of_x(x, A, lamda)
    sq_ave = (running_total / num_samples) ** 2

    return sum_of_sqs - sq_ave


test_lamdas = [i * 0.05 for i in range(1, 61)]
variances = []

for i, lamda in enumerate(test_lamdas):
    print("lambda {i + 1}/{len(test_lamdas)}: {lamda}")
    A = lamda
    variances.append(get_IS_variance(lamda, 10000))
    clear_output(wait=True)

optimal_lamda = test_lamdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print("Optimal Lambda: {optimal_lamda}")
print("Optimal Variance: {IS_variance}")
print("Error: {(IS_variance / 10000) ** 0.5}")