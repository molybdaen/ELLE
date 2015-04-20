__author__ = 'Johannes'

from compression.data import MovielensDataset
from compression.autoencoder import AutoEncoder
from matplotlib import pyplot
import numpy as np
import theano.tensor as T
from scipy.io import savemat
import sys
import operator
from sklearn.neural_network import BernoulliRBM

PATH_DATA_ROOT = r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m"
PATH_DATA_EVAL = r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m"

if __name__ == '__main__':

    codeLength = 5

    def getSummary(clusters, recs, k):
        attributes = ["M", "F", "Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+", "other", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist", "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]
        for (i, codeRec) in enumerate(recs):
            binStr = format(i, '00'+str(codeLength)+'b')
            kmaxvalindices = codeRec.argsort()[-k:][::-1]
            indices = list(np.asarray(kmaxvalindices, dtype=int))
            print " "
            print binStr
            for ind in indices:
                print "%s : %.3f" % (attributes[ind], codeRec[ind])


    def roundPlusMinus(x):
        x[x<0] = -1.
        x[x>=0] = 1.
        return x

    def plot_first_k_numbers(X,k):
        m=X.shape[0]
        k=min(m,k)
        j = int(round(k / 10.0))

        fig, ax = pyplot.subplots(j,10)

        for i in range(k):

            w=X[i,:]


            w=w.reshape(10,10)
            ax[i/10, i%10].imshow(w,cmap=pyplot.cm.gist_yarg,
                          interpolation='nearest', aspect='equal')
            ax[i/10, i%10].axis('off')


        pyplot.tick_params(\
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        pyplot.tick_params(\
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left='off',
            right='off',    # ticks along the top edge are off
            labelleft='off')

        fig.show()

    def m_test(data):
        activation_function = T.nnet.sigmoid
        output_function=T.nnet.sigmoid
        A = AutoEncoder(data, codeLength, activation_function, output_function)
        # great for squarederror: e=50, bathcsize=3, lr=0.3
        A.train(n_epochs=100, mini_batch_size=1, learning_rate=0.3)
        codes = A.get_hidden(data)
        return (A, codes)
        # W=np.transpose(A.get_weights()[0])
        # print np.shape(W)
        # plot_first_k_numbers(W, 100)
        # plot_first_k_numbers(W,100)

    def b_test(data):
        rbm = AutoEncoder(n_components=codeLength)
        rbm.fit(data)
        codes = rbm.transform(data)
        return codes

    data = MovielensDataset(PATH_DATA_ROOT)
    uamat = data.getUsersAttributesMatrix()
    print np.shape(uamat)
    # uimat = data.getUserItemMatrix()

    # print np.shape(uimat)
    matUamat = {"uamat": uamat}
    savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\usermat.mat", matUamat)

    sexMat = uamat[:,[0,1]]
    ageMat = uamat[:,[2,3,4,5,6,7,8]]
    occMat = uamat[:,range(9,30)]

    (encoder, codes) = m_test(uamat)
    # codes = b_test(uamat)

    matCodes = {"codes": codes}

    savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\codes.mat", matCodes)

    # user34 = data.getRatingsOfUser(34)
    # print len(user34)
    # for r in user34:
    #     print r
    #
    # minuser = 34
    # mindist = 1.0e+10
    # for (i,c) in enumerate(codes[0]):
    #     d = np.linalg.norm(codes[0][34] - c)
    #     if d < mindist and i is not 34:
    #         mindist = d
    #         minuser = i
    #     # if i > 1000:
    #     #     break
    #     # print ", ".join([str(cv) for cv in c])
    #
    # print "user with mindist %.4f: index %d" % (mindist, minuser)
    #
    # minuserratings = data.getRatingsOfUser(minuser)
    # print len(minuserratings)
    # for r in minuserratings:
    #     print r
    #
    # print "======================="
    # print "COMMMMOOOONN"
    # print ""
    # print "user34: %d rated movies" % (len(user34))
    # print "user%d: %d rated movies" % (minuser, len(minuserratings))
    # common = 0
    # for r in minuserratings:
    #     for r34 in user34:
    #         if r[0] == r34[0]:
    #             print "%d <-> %d :: %d - %s" % (r34[1], r[1], r[0][0], r[0][1])
    #             common += 1
    #             break
    #
    # print "Common movies:"
    # print common
    #
    # for (i,c) in enumerate(codes[0]):
    #     if i > 1000:
    #         break
    #     print ", ".join([str(cv) for cv in c])


    # Evaluate Clustering
    stats = []
    clusters = {}
    allUserIds = range(0, 6041)
    userClusters = np.zeros((np.shape(uamat)[0], codeLength))
    mclusts = {"uamat": uamat}
    for (userId,c) in enumerate(codes[0]):
        # minUser = -1
        # minDist = 1.0e+10
        # minCodeVec = None
        # minCodeStr = ""
        # for i in xrange(2**codeLength):
        #     binStr = format(i, '00'+str(codeLength)+'b')
        #     codeVec = np.asarray([float(bit) for bit in binStr])
        #     d = np.linalg.norm(c - codeVec)
        #     if d < minDist:
        #         minDist = d
        #         minUser = userId
        #         minCodeVec = codeVec
        #         minCodeStr = binStr
        # userClusters[userId] = minCodeVec
        roundedCluster = [x for x in np.round(c)]
        userClusters[userId] = roundedCluster
        roundedClusterStr = [str(int(x)) for x in roundedCluster]
        minCodeStr = "".join(roundedClusterStr)
        minUser = userId
        d = np.linalg.norm(c - np.round(c))

        if minCodeStr not in clusters:
            clusters[minCodeStr] = {"userIds": [minUser], "userDists": [d]}
        else:
            clusters[minCodeStr]["userIds"].append(minUser)
            clusters[minCodeStr]["userDists"].append(d)

    codevecs = np.zeros((2**codeLength, codeLength))
    for i in xrange(2**codeLength):
        binStr = format(i, '00'+str(codeLength)+'b')
        codeVec = np.asarray([float(bit) for bit in binStr])
        codevecs[i] = codeVec
        # codeVec = codeVec.reshape((1, len(codeVec)))
        # print str(codeVec)
        # print str(encoder.get_reconstruction(codeVec))
    recs = encoder.get_reconstruction(codevecs)

    savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\recs.mat", {"recs": recs})

    clusterSizes = []
    clusts = {}
    for c in clusters:
        mean = np.mean(np.asarray(clusters[c]["userDists"]))
        std = np.std(np.asarray(clusters[c]["userDists"]))
        clusterSizes.append(len(clusters[c]["userIds"]))
        print len(clusters[c]["userIds"])
        # print "%s : %d : mean %.3f : std %.3f : %s" % (c, len(clusters[c]["userIds"]), mean, std, ", ".join([str(data.users.getUser(u)) for u in clusters[c]["userIds"]]))
        # for u in clusters[c]["userIds"]:
        #     print str(data.users.getUser(u))

    # hiddens = encoder.get_hidden(uamat[clusters["00101"]["userIds"]])[0]
    # print np.shape(hiddens)
    # recons = encoder.get_reconstruction(hiddens)
    # print np.shape(recons)
    # clusts["a00101"] = recons
    npcs = np.asarray(clusterSizes)
    print "ClusterSize: mean %.2f std %.2f" % (np.mean(npcs), np.std(npcs))

    # getSummary(clusters, recs, 5)
    savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\W.mat", {"W": encoder.W.get_value()})
    # savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\ELLE\ml-1m\uaclusts.mat", clusts)

    savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\userclusts.mat", mclusts)

    transformedFromUser = encoder.get_reconstruction_error(uamat)
    transformedFromCode = encoder.get_reconstruction_error_from_code(uamat, userClusters)
    for c in clusters:
        print c
        for u in clusters[c]["userIds"]:
            print "%s : %.3f : %.3f" % (str(data.users.getUser(u)), transformedFromUser[2][u], transformedFromCode[1][u])