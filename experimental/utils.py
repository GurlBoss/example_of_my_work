import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

Gs_static = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
Gs_dynamic = ['pin', 'rotate', 'touch', 'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down']
Gs_actual = Gs_dynamic  # change for your actual gestures
work_dir = os.path.abspath(os.path.dirname(__file__))

class gesture():
    def __init__(self, name,id):
        self.name = name
        self.id = id
        self.gest_list = None

    def get_len(self):
        return len(self.gest_list)


def list_of_gestures(data, labels):
    num_of_Gs = int(max(labels) + 1)
    tmp_list = []
    for i in range(num_of_Gs):
        tmp_gs = gesture(None,i)
        tmp_gs.gest_list = [data[j, ...] for j in range(len(labels)) if labels[j] == i]
        tmp_list.append(tmp_gs)
    return tmp_list

def gest_to_single_list(data):
    tmp_classes = []
    tmp_gestures = []
    for object in data:
        for single_gest in object.gest_list:
            tmp_classes.append(object.id)
            tmp_gestures.append(single_gest)
    return tmp_gestures,tmp_classes

def create_val_data(data,classes,percent = 30):
    main_list = list_of_gestures(data, classes);
    val_data,val_rec = [],[]
    percent = 100-percent
    for object in main_list:
        percent1 = percent / 100
        c = round(len(object.gest_list) * percent1)
        for i in range(c):
            tmp_timeseries = object.gest_list.pop(random.randrange(len(object.gest_list)))
            val_data.append(tmp_timeseries)
            val_rec.append(object.id)
    train_data, train_rec = gest_to_single_list(main_list)
    return val_data,val_rec,train_data,train_rec

def pca_calculate(path):
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    f.close()
    for sample in data:
        pca_2 = PCA(n_components=1)
        pca_2.fit(sample)
        new_data = pca_2.transform(sample)
        plt.plot(new_data)
    plt.show()
    #sns.scatterplot(x = data_pca_2[:,0],y = data_pca_2[:,1],hue = acc.accuracy,palette = "OrRd")
    #ariability = np.cumsum(pca_4.explained_variance_ratio_ * 100)
    plt.title("2D Scatterplot: 59.6% of the variability captured")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig(work_dir + "/output/PCA.eps")
    plt.savefig(work_dir + "/output/PCA.png")
    print("hello")

def grid_search_graph(path,s = "",type = ""):
    data = pd.read_csv(path, usecols=[2, 3, 4, 5])
    data1 = data[data["hidden size"] == 32]
    layrs = [2,4]
    hidden = [32,64,128,256,512]
    colors = ['b','r','c','m','g']
    l = []
    for lar in layrs:
        i=0
        for hid in hidden:
            data1 = data[data.layers == lar]
            data1 = data1[data1["hidden size"] == hid]
            plt.xscale('log')
            if lar == 4:
                ls = '--'
            else:
                ls = '-'
            l1, = plt.plot(data1.lr, data1.accuracy,ls=ls,color = colors[i])
            l.append(l1)
            plt.ylim([0,1])
            i= i+ 1
    legend1 = plt.legend( [32,64,128,256,512],loc = 3,title = "Hidden size",ncol = 2)
    k = [l[0],l[5]]
    plt.legend(k,[2,4],loc = 4,title = "Layers")
    plt.gca().add_artist(legend1)
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Learning rate")
    plt.savefig(work_dir+"/output/pics/grid_search_"+str(type)+"_"+str(s)+".png")
    print("done")
    plt.clf()

def grid_search_graph_avg(path,s = "",type = ""):
    data = pd.read_csv(path[0], usecols=[2, 3, 4, 5])
    data1 = pd.read_csv(path[1], usecols=[2, 3, 4, 5])
    data2 = pd.read_csv(path[2], usecols=[2, 3, 4, 5])
    data3 = pd.read_csv(path[3], usecols=[2, 3, 4, 5])
    data4 = pd.read_csv(path[4], usecols=[2, 3, 4, 5])
    layrs = [2,4]
    hidden = [32,64,128,256,512]
    colors = ['b','r','c','m','g']
    l = []
    for lar in layrs:
        i=0
        for hid in hidden:
            data_ = data[data.layers == lar]
            data_ = data_[data_["hidden size"] == hid]
            data_1 = data1[data1.layers == lar]
            data_1 = data_1[data_1["hidden size"] == hid]
            data_2 = data2[data2.layers == lar]
            data_2 = data_2[data_2["hidden size"] == hid]
            data_3 = data3[data3.layers == lar]
            data_3 = data_3[data_3["hidden size"] == hid]
            data_4 = data4[data4.layers == lar]
            data_4 = data_4[data_4["hidden size"] == hid]
            plt.xscale('log')
            if lar == 4:
                ls = '--'
            else:
                ls = '-'
            a0 = np.expand_dims(data_.accuracy, axis=1)
            a1 = np.expand_dims(data_1.accuracy, axis=1)
            a2 = np.expand_dims(data_2.accuracy, axis=1)
            a3 = np.expand_dims(data_3.accuracy, axis=1)
            a4 = np.expand_dims(data_4.accuracy, axis=1)
            acc = np.concatenate((a0,a1,a2,a3,a4),axis = 1)
            acc_max = np.amax(acc, axis=1)
            acc_min = np.amin(acc, axis=1)
            acc_mean = np.mean(acc,axis = 1)
            l1, = plt.plot(data_.lr, acc_mean,ls=ls,color = colors[i])
            l2 = plt.plot(data_.lr, acc_max, ls=ls, color=colors[i],alpha = 0.2,label = "max/min")
            l3 = plt.plot(data_.lr, acc_min, ls=ls, color=colors[i],alpha = 0.2)
            l.append(l1)
            plt.ylim([0,1])
            i= i+ 1
    k = l[0:5]
    legend1 = plt.legend( k,[32,64,128,256,512],loc = 3,title = "Hidden size",ncol = 2)
    k = [l[0],l[5]]
    plt.legend(k,[2,4],loc = 4,title = "Layers")
    plt.gca().add_artist(legend1)
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Learning rate")
    plt.savefig(work_dir+"/output/pics/avg_grid_search_"+str(type)+"_"+str(s)+".jpg",dpi=300)
    print("done")
    plt.clf()

if __name__ == '__main__':
    #pca_calculate(work_dir+"/dataset_hand_gest/train/hand_70_train.npy")
    #grid_search_graph(work_dir+"/output/copelia/diff_tip_shape_gru_grid_search.csv")
    #datasets = ["90", "60", "70", "80", "50"]
    types = ["lstm","gru"]
    #

    datasets = ["fft"]
    for dataset in datasets:
        for type in types:
            if type=="gru":
                grid_search_graph(work_dir+"/output/audio/grid_search/"+dataset+"_"+type+"_grid_search.csv",type = type,s = dataset)

    datasets = ["forces", "shape_ori", "shape_ori_plus_pos", "tip_minus_shape_pos", "shape_pos", "tip_pos"]
    for dataset in datasets:
        for type in types:
            grid_search_graph(
                work_dir + "/output/copelia/grid_search/" + dataset + "_" + type + "_grid_search.csv", type=type,
                s=dataset)

    datasets = ["90", "60", "70", "80", "50"]
    for dataset in datasets:
        for type in types:
            grid_search_graph(
                work_dir + "/output/hand/" + dataset + "_" + type + "_grid_search.csv", type=type,
                s=dataset)

    datasets = ["original"]
    for dataset in datasets:
        for type in types:
            grid_search_graph(
                work_dir + "/output/moves/" + dataset + "_" + type + "_grid_search.csv", type=type,
                s=dataset)
            '''
            path = [work_dir + "/output/moves/grid_search/" + dataset + "_" + type + "_grid_search.csv",
                    work_dir + "/output/moves/grid_search/" + dataset + "_" + type + "_grid_search1.csv",
                    work_dir + "/output/moves/grid_search/" + dataset + "_" + type + "_grid_search2.csv",
                    work_dir + "/output/moves/grid_search/" + dataset + "_" + type + "_grid_search3.csv",
                    work_dir + "/output/moves/grid_search/" + dataset + "_" + type + "_grid_search4.csv"]
            grid_search_graph_avg(path,
                              type=type, s=dataset)
            '''