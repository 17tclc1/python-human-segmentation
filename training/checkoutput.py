import pickle

if __name__ == '__main__':
    a_knownNames = 'output/knownNames.pickle'
    a_knownEmbeddings = 'output/knownEmbeddings.pickle'
    a_z = 'output/embeddings.pickle'

    t_knownNames = pickle.loads(open(a_knownNames, "rb").read())
    t_knownEmbeddings = pickle.loads(open(a_knownEmbeddings, "rb").read())

    t_z = pickle.loads(open(a_knownNames, "rb").read())
    list = []
    for i in t_z:
        if i not in list:
            list.append(i)
            print(i)
        else:
            continue