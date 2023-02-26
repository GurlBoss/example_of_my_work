import numpy as np
from matplotlib import pyplot as plt
import math
from tqdm import tqdm
from matplotlib import rcParams
import os

work_path = os.path.abspath(os.path.dirname(__file__))
Gs_dynamic = ['pin', 'rotate', 'touch', 'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down']
data_labels = Gs_dynamic


# counting DTW distance
# could be modified to **
def dtw(s, t,all = None,actual= None):
    DTW = np.full((len(s) + 1, len(t) + 1), np.inf)
    DTW[0, 0] = 0
    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            l = s[i - 1]
            p = t[j - 1]
            cost = np.linalg.norm(abs(p - l))  # euclidean distance
            DTW[i, j] = cost + min(DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j])
            # modification:
            # DTW[i, j] = cost**2 + min(DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j])
    if all and actual and actual % 100 == 0:
        print("%5d/%5d" % (actual,all))
    return DTW[len(s), len(t)]
    # modification:
    # return math.sqrt(DTW[len(s), len(t)])


# inspiration:
# https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
def lcss(x, y):
    m = len(x)
    n = len(y)
    L = np.full((m + 1, n + 1), np.inf)
    threshold = 1
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i, j] = 0
            elif np.linalg.norm(abs(x[i - 1] - y[j - 1])) < threshold:
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = max(L[i - 1, j], L[i, j - 1])
    return L[m, n]


# TODO check with the teacher if the implementation is alright
def edr(x, y):
    m = len(x)
    n = len(y)
    threshold = 1
    E = np.full((m + 1, n + 1), np.inf)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                E[i, j] = j
            elif j == 0:
                E[i, j] = i
            else:
                dist = np.linalg.norm(abs(x[i - 1] - y[j - 1]))

                # check with threshold
                if dist < threshold:
                    d = 0
                else:
                    d = 1
                E[i, j] = min(E[i - 1, j - 1] + d, E[i - 1, j] + 1,
                              E[i, j - 1] + 1)
    return E[m, n]

# counting averages DTW dist for all classes
def average(x, y):
    classes = int(max(y) + 1)
    avg = [0 for i in range(classes)]
    sum = [0 for i in range(classes)]
    for i in range(len(x)):
        if not math.isinf(x[i]):
            avg[int(y[i])] += x[i]
            sum[int(y[i])] += 1
    for i in range(classes):
        avg[i] = avg[i] / sum[i]
    return avg


# Create graph from x,y axis for distances
def plot_graph(x, y, title="", xlabel="", ylabel="", file_name='graph'):
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y, label="stars", c=y,
                marker=".", s=20)

    x2 = average(x, y)
    y2 = [int(i) for i in range(len(x2))]

    plt.scatter(x2, y2, label="stars", c='red',
                marker="x", s=30)
    #plt.yticks(np.arange(len(data_labels)), data_labels)
    rcParams.update({'figure.autolayout': True})
    plt.savefig(work_path+'/output/'+file_name + '.png', format='png', bbox_inches="tight")


def basic_graph_dtw(x, y,file_name):
    tmp_arr = []
    first_data = x[0]
    tmp_point = first_data
    bar = tqdm(total=len(y))
    bar.set_description("Computing dtw")
    for i in range(len(y)):
        bar.update(1)
        tmp_arr.append(dtw(tmp_point, x[i]))
    plot_graph(tmp_arr, y, title="DTW distance", xlabel="Distance"
               , ylabel="Gesture label", file_name=file_name)


def basic_graph_lcss(x, y,file_name):
    tmp_arr = []
    first_data = x[0]
    tmp_point = 0*first_data
    bar = tqdm(total=len(y))
    bar.set_description("Computing LCSS")
    for i in range(len(y)):
        bar.update(1)
        tmp_arr.append(lcss(tmp_point, x[i]))
    plot_graph(tmp_arr, y, title="LCSS distance, epsilon = 1", xlabel="Distance"
               , ylabel="Gesture label", file_name=file_name)


def basic_graph_edr(x, y,file_name):
    tmp_arr = []
    first_data = x[0]
    tmp_point = first_data
    bar = tqdm(total=len(y))
    bar.set_description("Computing edr")
    for i in range(len(y)):
        bar.update(1)
        tmp_arr.append(edr(tmp_point, x[i]))
    plot_graph(tmp_arr, y, title="EDR distance, epsilon = 1", xlabel="Distance"
               , ylabel="Gesture label", file_name=file_name)
