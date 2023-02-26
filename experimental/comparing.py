import numpy as np
import os
import distance
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import main
from joblib import Parallel, delayed
import multiprocessing
import statistics

# Delcare Gs for import data
Gs_static = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
Gs_dynamic = ['pin', 'rotate', 'touch', 'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down']
Gs_dynamic2 = ['pin', 'rotate', 'touch', 'swipe left', 'swipe right', 'swipe up', 'swipe down']
audio = ['I','Z','P','D','O']
data_labels = audio
Gs_actual = Gs_dynamic  # change for your actual gestures
# Path to the dataset default set to
# /home/<user>/<workspace>/src/mirracle_gestures/include/data/learning/
PATH = os.path.abspath(os.path.dirname(__file__)) + "/Dataset/learning/"
work_dir = os.path.abspath(os.path.dirname(__file__))
''' DTW Algorithm
More info here: https://en.wikipedia.org/wiki/Dynamic_time_warping '''


class gesture():
    def __init__(self, name):
        self.name = name
        self.gest_list = None

    def get_len(self):
        return len(self.gest_list)


def list_of_gestures(data, labels):
    num_of_Gs = int(max(labels) + 1)
    tmp_list = []
    for i in range(num_of_Gs):
        tmp_gs = gesture(Gs_actual[i])
        tmp_gs.gest_list = [data[j, :, :] for j in range(len(labels)) if labels[j] == i]
        tmp_list.append(tmp_gs)
    return tmp_list


# count distance matrix for data
# data and rec have to be in "list" as import_data return

def dist_matrix_dtw(data, rec,dataset = ''):
    n = len(rec)
    dist_matrix = np.empty([n, n])
    n2 = 0
    length = (n*n)/2
    c = 0
    num_cores = multiprocessing.cpu_count()
    for i in range(n):

        start = time.time()
        dist_matrix[i, n2:] = Parallel(n_jobs=num_cores) \
            (delayed(distance.dtw)(data[i].astype(np.double),
                                   data[j].astype(np.double),
                                   all = length,actual = c + j) for j in range(n2, n))
        end = time.time()
        one_sample = (end - start) / 60 / 60 / (n-i)
        print("%5d / %5d Predicted time %.2f h" % (c, length, (length - c) * one_sample))
        '''
        for j in range(n2,n):
            start = time.time()
            data[i]=data[i].astype(np.double)
            data[j]=data[j].astype(np.double)
            dist_matrix[i, j] = distance.dtw(data[i], data[j])

            end = time.time()
            c = c+1
            one_sample = (end-start)/60/60
            print("%5d / %5d Predicted time %d h" % (c,length,(length-c)*one_sample))
        '''
        n2 = n2 + 1
        c = c+ (n-n2)

    with open(work_dir+'/output/data/'+'dtw_dist_matrix_'+dataset+'.npy', 'wb') as f:
        np.save(f, dist_matrix)
    f.close()
    return dist_matrix

def dist_matrix_lcss(data, rec):
    n = len(rec)
    dist_matrix = np.empty([n, n])
    pbar = tqdm(total=n * n)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = distance.lcss(data[i], data[j])
            pbar.update(1)
    with open('lcss_dist_matrix_audio.npy', 'wb') as f:
        np.save(f, dist_matrix)
    f.close()
    return dist_matrix

def dist_matrix_edr(data, rec):
    n = len(rec)
    dist_matrix = np.empty([n, n])
    pbar = tqdm(total=n * n)
    n2 = n
    for i in range(n):
        for j in range(n2):
            data[i]=data[i].astype(np.double)
            data[j]=data[j].astype(np.double)
            dist_matrix[i, j] = distance.edr(data[i], data[j])
            pbar.update(1)
        #n2 = n2 - 1
    with open('edr_dist_matrix_audio.npy', 'wb') as f:
        np.save(f, dist_matrix)
    f.close()
    return dist_matrix

def basic_graphs(x, y):
    distance.basic_graph_edr(x, y,"edr_audio")
    distance.basic_graph_dtw(x, y,"dtw_audio")
    distance.basic_graph_lcss(x, y,"lcss_audio")

def conf_matrix(path_open,path_save,labels,lengths):
    with open(path_open,'rb') as f:
        dist_array = np.load(f)
    plt.clf()
    array_int = dist_array.astype(int)
    dist_array = dist_array.astype(int)
    dist_array[dist_array<0] = 0
    dist_array[dist_array > 50] = 50
    dist_array = dist_array+dist_array.T-np.diag(np.diag(dist_array))
    plt.imshow(dist_array, cmap='hot')
    data_labels = labels
    plt.colorbar(label="Distance")
    plt.yticks(lengths, data_labels)
    plt.xticks(lengths, data_labels,rotation=0)
    plt.savefig(path_save, format='eps', bbox_inches="tight")

if __name__ == "__main__":
    datasets = ["forces","shape_ori_plus_pos","shape_ori","shape_pos","tip_minus_shape_pos","tip_pos"]
    #datasets = ["fft","max"]
    #datasets = ["original"]
    #lab = ["dance", "fly", "wave"]
    #lab = ['P','D','I','Z','O']
    lab = ["push", "bump", "lift"]
    #lab = ["label 1", "label 2", "label 3"]
    #lenghts = [50,150,250,350,450]
    #lenghts = [10,30,50]
    lenghts = [50,150,250]
    only_conf = True
    knn=True
    for dataset in datasets:
        if knn:
            train_data, train_rec, val_data, val_rec = main.open_dir_copelia(dataset)
            main.conf_mat(lab,train_data, train_rec, val_data, val_rec, dataset, rot=0)
        else:
            if not only_conf:
                train_data, train_rec, val_data, val_rec = main.open_dir_copelia(dataset)
                dist_array = dist_matrix_dtw(train_data,train_rec,dataset = dataset)
            conf_matrix(path_open=work_dir + '/output/data/' + 'dtw_dist_matrix_' + dataset + '.npy',
                        path_save=work_dir + '/output/pics/' + 'dtw_dist_matrix_' + dataset + '.eps',
                        labels=lab,
                        lengths=lenghts)
    quit()

    '''

    quit()

    x = a
    y = rec
    distance.basic_graph_dtw(x, y)
    quit()


    dist_array = dist_matrix_dtw(a,rec)
    c= ssd.squareform(dist_array)
    z = linkage(dist_array, method='complete')
    fig = plt.figure(figsize=(25, 10))
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(z,
               #p = 7,
               #truncate_mode = 'lastp',
               color_threshold=120,
               labels=rec)
    numclust = 7
    fl = fcluster(z, numclust, criterion='maxclust')

    fig.savefig("final_DXPALM.png", format='png')
    fig.savefig("final_DXPALM.eps", format='eps')

    
    print("Loading data")
    data = import_data.import_data(learn_path=PATH, Gs=Gs_dynamic)
    print("Data loaded")
    rec = data['dynamic']['Y']

    # Cartesian palm coordinates (shape = [Recordings x Time x 4]), [X,Y,Z,(euclidean distance from palm to point
    # finger tip)]
    xpalm = data['dynamic']['Xpalm']

    # Cartesian palm velocities (shape = [Recordings x Time x 4])
    dxpalm = data['dynamic']['DXpalm']

    X = data['static']['X']
    # Flags (gesture IDs) (shape = [Recordings])
    
    
    dist_array = ssd.squareform(xpalm,rec)
    c= ssd.squareform(dist_array)
    z = linkage(dist_array,method='average')
    fig = plt.figure(figsize=(25, 10))
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(z,
               #p = 7,
               #truncate_mode = 'lastp',
               color_threshold=120,
               labels=rec)
    numclust = 7
    fl = fcluster(z, numclust, criterion='maxclust')

    fig.savefig("final_DXPALM.png", format='png')
    fig.savefig("final_DXPALM.eps", format='eps')

   



    # plot basic graphs
    # basic_graphs(xpalm, rec)

    dtw_dist_mat = dist_matrix_dtw(xpalm, rec)
    dist_array = ssd.squareform(dtw_dist_mat)

    Experiments:
    with open('rocords.npy', 'rb') as f:
    rec= np.load(f)
    f.close()
    with open('test.npy', 'rb') as f:
        a=np.load(f)
    f.close()
    plt.imshow(a.astype(int), cmap='hot')
    plt.colorbar()
    plt.savefig('dtw_dist_matrix.png', format='png', bbox_inches="tight")
    A = np.array([[1, 2, 5], [3, 4, 6]])
    c = A[1, :]
    r = A[:, 1]
    K = np.zeros((2, 5))
    A = 1 / A
    A1 = np.sum(A, axis=0)
    A2 = np.sum(A, axis=1)
    timeseries = np.array([[1,2,2,3],[2,3,5,4]],dtype=np.double)
    timeseries = dxpalm.astype(np.double)

    model3 = clustering.LinkageTree(dtw_ndim.distance_matrix_fast, {'ndim':4})
    cluster_idx = model3.fit(timeseries)
    fig = plt.figure(figsize=(25, 10))
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(cluster_idx)
    fig.savefig("final_DXPALM.png",format='png')
    
    model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
    cluster_idx = model2.fit(timeseries)
    model2.plot("hihi.eps")
    
    model = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=7)
    cluster_idx = model.fit(timeseries)
    model.plot(filename="experiment.png", axes=None, ts_height=100,
               bottom_margin=0, top_margin=1, ts_left_margin=0, ts_sample_length=1,
               tr_label_margin=3, tr_left_margin=2, ts_label_margin=0, show_ts_label=True,
               show_tr_label=True, cmap='viridis_r', ts_color=None)
    
    gs_in_list = list_of_gestures(xpalm, data['dynamic']['Y'])
    dtw_for_xpalm(gs_in_list, rec)
    
    model = clustering.kmeans.KMeans(k=7, max_it=10, max_dba_it=10, dists_options={"window": 1})
    cluster_idx, performed_it = model.fit(timeseries, use_c=True, use_parallel=False)
    model.plot(filename="experiment2.eps", axes=None, ts_height=10, bottom_margin=2, top_margin=2, ts_left_margin=0, ts_sample_length=1,
         tr_label_margin=3, tr_left_margin=2, ts_label_margin=0, show_ts_label=None, show_tr_label=None,
         cmap='viridis_r', ts_color=None)
    '''
