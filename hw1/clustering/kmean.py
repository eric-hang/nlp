from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import gensim
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

# use num_point to show the
num_point = 50


def get_key(list, value):
    for i in range(len(list)):
        times = 10
        for j in range(10):
            if abs(list[i][j]-value[j]) < 0.01:
                times -= 1
            if times == 0:
                return key
    return 0


def cluster():
    # load the model
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/chenyiwei/hws/Assignment/dataset/wiki_vectors_word2vec.txt', binary=False)
    keys = model.wv.vocab.keys()
    with open('./word_list', 'rb') as f:
        word_list = pickle.load(f)
    # load wordvector for each key in the model
    dataset = []
    for word in word_list:
        dataset.append(model.wv[word])
    dataset_embedded = TSNE(n_components=2, early_exaggeration=6).fit_transform(dataset)
    # for diff num of clusters, to do the cluster algorithm
    for num_cluster in range(2, 8):
        # do the kmeans algorithm
        km_cluster = KMeans(n_clusters=num_cluster, max_iter=1000, init='k-means++')
        km_cluster.fit(dataset)
        # start draw the image
        cents = km_cluster.cluster_centers_  # the centroid
        '''for cent in cents:
            idx = get_key(dataset, cent)
            print(idx)
            plt.scatter(dataset_embedded[idx][0], dataset_embedded[idx][1], c='r', marker='*')'''
        labels = km_cluster.labels_
        mark = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        j = 0   # to count the point in the dataset
        '''flag = 0   # to judge which cluster should the point belog
        n_p = 0    # to count how many point already been drew
        for label in labels:
            if flag == num_cluster:
                break
            if label == flag:
                plt.scatter(dataset_embedded[j][0], dataset_embedded[j][1], c=mark[label])
                n_p += 1
            if n_p == num_point:
                n_p = 0
                flag += 1
            j += 1'''
        for label in labels:
            plt.scatter(dataset_embedded[j][0], dataset_embedded[j][1], c=mark[label])
            j += 1
        plt.title('k-means '+str(num_cluster)+' clusters', fontsize=20)
        plt.savefig('./pub_image/' + str(num_cluster) + '_cluster1.jpg')
        plt.show()


if __name__ == '__main__':
    cluster()
