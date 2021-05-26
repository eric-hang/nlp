from sklearn.cluster import KMeans
import gensim
import numpy
import time
import matplotlib.pyplot as plt


def cluster():
    # load the model
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/chenyiwei/hws/Assignment/dataset/pubmed_vectors_word2vec.txt')
    keys = model.wv.vocab.keys()
    # load wordvector for each key in the model
    dataSet = []
    for key in keys:
        dataSet.append(model[key])
    # for diff k, to do the cluster algorithm
    for k in range(5, 10):
        clf = KMeans(n_clusters=k)
        clf.fit(dataSet)
        cents = clf.centroids  # 质心
        labels = clf.labels  # 样本点被分配到的簇的索引
        sse = clf.sse
        # 画出聚类结果，每一类用一种颜色
        colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
        # n_clusters = 10
        for i in range(k):
            index = np.nonzero(labels == i)[0]
            x0 = X[index, 0]
            x1 = X[index, 1]
            y_i = y[index]
            for j in range(len(x0)):
                plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i], fontdict={'weight': 'bold', 'size': 9})
            plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i], linewidths=12)
        plt.title("SSE={:.2f}".format(sse))
        plt.axis([-30, 30, -30, 30])
        plt.savefig('./'+str(k)+'cluster.jpg')
        plt.show()


if __name__ == '__main__':
    cluster()
