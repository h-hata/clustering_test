import pickle


def load2D(fname):
    with open(fname, "rb") as f:
        clust1 = pickle.load(f)
        clust2 = pickle.load(f)
        labels = pickle.load(f)
    return (clust1, clust2, labels)
