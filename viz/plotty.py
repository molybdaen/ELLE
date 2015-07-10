__author__ = 'johannesjurgovsky'

import cPickle
import matplotlib.pyplot as plt
from utils import utils
from config import Config
import numpy as np
import scipy
from ellearning.ELAlgorithm import Autoencoder
import numpy.fft as fft


def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class Plotty(object):

    def __init__(self, model, logs):

        self.model = model
        self.logs = logs
        self.fAutoencoder = False

        if type(model) == type(Autoencoder):
            self.fAutoencoder = True

    def compareElasticCompound(self, elastic_logs, compound_logs):

        e_scores_f1, c_scores_f1 = [], []
        e_pres, c_pres = [], []
        e_recs, c_recs = [], []
        e_times, c_times = [], []
        e_times_integrated = [0.]
        e_errors, c_errors = [], []
        e_errors_sep, c_errors_sep = [], []
        e_last_errors, c_last_errors = [], []

        e_tmp_list = []
        for k in elastic_logs:
            if representsInt(k):
                e_tmp_list.append(int(k))
        c_tmp_list = []
        for k in compound_logs:
            if representsInt(k):
                c_tmp_list.append(int(k))

        e_num_tests = max(e_tmp_list)
        c_num_tests = max(c_tmp_list)

        for numNodes in range(1, e_num_tests+1):
            (pre, rec, f1) = elastic_logs[str(numNodes)][Autoencoder.STR_SCORES]
            time = elastic_logs[str(numNodes)][Autoencoder.STR_TIMES]
            error = elastic_logs[str(numNodes)][Autoencoder.STR_ERRORS]

            e_scores_f1.append(np.mean(f1))
            e_pres.append(np.mean(pre))
            e_recs.append(np.mean(rec))
            e_errors += error
            e_errors_sep.append(error)
            e_last_errors.append(error[-1])
            e_times += time
            e_times_integrated.append(e_times_integrated[-1]+sum(time))

        for numNodes in range(1, c_num_tests+1):
            (pre, rec, f1) = compound_logs[str(numNodes)][Autoencoder.STR_SCORES]
            time = compound_logs[str(numNodes)][Autoencoder.STR_TIMES]
            error = compound_logs[str(numNodes)][Autoencoder.STR_ERRORS]

            c_scores_f1.append(np.mean(f1))
            c_pres.append(np.mean(pre))
            c_recs.append(np.mean(rec))
            c_errors += error
            c_errors_sep.append(error)
            c_last_errors.append(error[-1])
            c_times += time

        plt.figure(1)
        (e_lpre, c_lpre, e_lrec, c_lrec, e_lf1, c_lf1) = plt.plot(e_pres, 'wo', c_pres, 'ko', e_recs, 'w^', c_recs, 'k^', e_scores_f1, 'b.', c_scores_f1, 'r.')
        plt.setp(e_lpre, label='Precision (E)')
        plt.setp(c_lpre, label='Precision (C)')
        plt.setp(e_lrec, label='Recall (E)')
        plt.setp(c_lrec, label='Recall (C)')
        plt.setp(e_lf1, label='F1 (E)')
        plt.setp(c_lf1, label='F1 (C)')
        plt.axis([0, 100, 0, 1])
        plt.grid(True)
        plt.title("MNIST 10-Digit Classification")
        plt.xlabel('Hidden Nodes')
        plt.ylabel('Mean Classification Scores')
        plt.legend(loc='lower right')
        plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "f1.pdf"))

        plt.figure(2)
        print len(e_last_errors)
        print len(c_last_errors)
        print e_last_errors
        print c_last_errors
        plt.step(range(1, e_num_tests+1), e_last_errors, 'b')
        plt.step(range(1, c_num_tests+1), c_last_errors, 'r')
        # plt.step(range(1,min((e_num_tests, c_num_tests))), c_last_errors)
        plt.grid(True)
        plt.legend(("Elastic", "Compound"))
        plt.xlabel('Hidden Layer Size')
        plt.ylabel('Reconstruction Error')
        plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "errors.pdf"))


        plt.figure(3, figsize=(20,6))
        ax = plt.subplot(2,1,1)
        summer = 0
        ticks = [summer]
        for (i, e) in enumerate(e_errors_sep):
            plt.plot(range(summer, summer+len(e)), e, 'b')
            # plt.plot((summer, summer), (0, 250), 'k-', linewidth=0.5, alpha=0.2)
            summer += len(e)
            ticks.append(summer)
        plt.grid(True)
        # Set the ticks and labels...
        labels = []
        for i in range(1, len(ticks)+1):
            if i % 10 == 0:
                labels.append(str(i))
            else:
                labels.append(" ")
        plt.xticks(ticks, labels)
        plt.legend(("Elastic",))
        plt.xlabel('Hidden Layer Size')
        plt.ylabel('Reconstruction Error')

        ax = plt.subplot(2,1,2)
        summer = 0
        ticks = [summer]
        for (i, e) in enumerate(c_errors_sep):
            plt.plot(range(summer, summer+len(e)), e, 'r')
            # plt.plot((summer, summer), (0, 250), 'k-', linewidth=0.5, alpha=0.2)
            summer += len(e)
            ticks.append(summer)
        plt.grid(True)
        # Set the ticks and labels...
        labels = []
        for i in range(1, len(ticks)+1):
            if i % 10 == 0:
                labels.append(str(i))
            else:
                labels.append(" ")
        plt.xticks(ticks, labels)
        plt.legend(("Compound",))
        plt.xlabel('Hidden Layer Size')
        plt.ylabel('Reconstruction Error')
        plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "error-drops.pdf"))

        plt.figure(4)
        plt.subplot(2,1,1)
        plt.plot(e_times)
        plt.plot(e_times, 'bo')
        plt.grid(True)
        plt.title("Elastic")
        plt.xlabel('Hidden Layer Size')
        plt.ylabel('Training Time [s] (per node)')

        plt.subplot(2,1,2)
        plt.plot(e_times_integrated, 'b_', c_times, 'r_')
        plt.grid(True)
        plt.legend(("Elastic", "Compound"), loc='upper left')
        plt.xlabel('Hidden Layer Size')
        plt.ylabel('Training Time [s] (total)')
        plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "times.pdf"))
        plt.show()



#######################
#### START Figures ####
#######################

# ELASTIC = "elastic"
# COMPOUND = "compound"
# SCORE = "score"
# TIME = "time"
# ERROR = "error"
#
# elastic_results = cPickle.load(open(utils.get_full_path(Config.PATH_EVAL_ROOT, (r"%s-results.pkl" % ELASTIC)), "r"))
# compound_results = cPickle.load(open(utils.get_full_path(Config.PATH_EVAL_ROOT, (r"%s-results-final.pkl" % COMPOUND)), "r"))
#
# e_scores_f1, c_scores_f1 = [], []
# e_pres, c_pres = [], []
# e_recs, c_recs = [], []
# e_times, c_times = [], []
# e_times_integrated = [0.]
# e_errors, c_errors = [], []
# e_errors_sep, c_errors_sep = [], []
# e_last_errors, c_last_errors = [], []
#
# e_num_tests = max([int(k) for k in elastic_results])
# c_num_tests = max([int(k) for k in compound_results])
#
# print "Maximum Layer Size: Elastic %d, Compound %d" % (e_num_tests, c_num_tests)
#
# for numNodes in range(1, e_num_tests+1):
#     (pre, rec, f1) = elastic_results[str(numNodes)][SCORE]
#     time = elastic_results[str(numNodes)][TIME]
#     error = elastic_results[str(numNodes)][ERROR]
#
#     e_scores_f1.append(np.mean(f1))
#     e_pres.append(np.mean(pre))
#     e_recs.append(np.mean(rec))
#     e_errors += error
#     e_errors_sep.append(error)
#     e_last_errors.append(error[-1])
#     e_times.append(time)
#     e_times_integrated.append(e_times_integrated[-1]+time)
#
# for numNodes in range(1, c_num_tests+1):
#     (pre, rec, f1) = compound_results[str(numNodes)][SCORE]
#     time = compound_results[str(numNodes)][TIME]
#     error = compound_results[str(numNodes)][ERROR]
#
#     c_scores_f1.append(np.mean(f1))
#     c_pres.append(np.mean(pre))
#     c_recs.append(np.mean(rec))
#     c_errors += error
#     c_errors_sep.append(error)
#     c_last_errors.append(error[-1])
#     c_times.append(time)



# n_groups = 10
#
# fig, ax = plt.subplots()
#
# index = np.arange(n_groups)
# bar_width = 0.35
#
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
#
# rects1 = plt.bar(index, compound, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  error_kw=error_config,
#                  label='Compound')
#
# rects2 = plt.bar(index + bar_width, elastic, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  error_kw=error_config,
#                  label='Elastic')
#
# plt.xlabel('Digit Group')
# plt.ylabel('F1-Score')
# plt.title('F1-Scores on MNIST by Digit Groups and Method')
# plt.xticks(index + bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
# plt.legend()
#
# plt.tight_layout()
# plt.show()

#####################
#### END Figures ####
#####################

def pickleFromWindowsEncoding(filename):
    newfile = filename + "-formatted.pkl"
    newFileH = open(newfile, 'wb')
    for l in open(filename, 'rb'):
        s = l.replace("\r\n", "\n")
        newFileH.write(s)
    newFileH.close()
    return cPickle.load(open(newfile, "rb"))

def extractInfo(logs):
    e_tmp_list = [int(k) for k in logs if representsInt(k)]
    e_num_tests = max(e_tmp_list)

    layer_sizes = []
    scores_f1, pres, recs = [], [], []
    errors, errors_sep, last_errors = [], [], []
    times, times_integrated, times_all, times_overall = [], [0.], [], []
    prev_errors, prev_times = [], []

    for nodeIdx in range(0, e_num_tests+1):
        if str(nodeIdx) in logs:
            prev_errors += logs[str(nodeIdx)][Autoencoder.STR_ERRORS]
            prev_times += logs[str(nodeIdx)][Autoencoder.STR_TIMES]
            times_overall += logs[str(nodeIdx)][Autoencoder.STR_TIMES]
            if Autoencoder.STR_SCORES in logs[str(nodeIdx)]:
                layer_sizes.append(nodeIdx+1)
                (pre, rec, f1) = logs[str(nodeIdx)][Autoencoder.STR_SCORES]
                time = logs[str(nodeIdx)][Autoencoder.STR_TIMES]
                error = logs[str(nodeIdx)][Autoencoder.STR_ERRORS]

                times.append(sum(time))
                times_all.append(time)

                scores_f1.append(np.mean(f1))
                pres.append(np.mean(pre))
                recs.append(np.mean(rec))

                errors += error
                errors_sep.append(prev_errors)
                prev_errors = []
                last_errors.append(error[-1])

                times_integrated.append(times_integrated[-1]+np.sum(prev_times))
                prev_times = []

    return layer_sizes, scores_f1, pres, recs, errors_sep, times, times_integrated[1:], times_all, times_overall

def plotScores(figIdx, dataset, e_layer_sizes, e_pres, e_recs, e_scores_f1, c_layer_sizes, c_pres, c_recs, c_scores_f1):
    plt.figure(figIdx)

    plt.subplot(3,1,1)
    plt.title("%s - Classification (Mean)" % dataset)
    # (e_lpre, c_lpre, e_lrec, c_lrec, e_lf1, c_lf1) = plt.plot(e_layer_sizes, e_pres, 'wo', c_layer_sizes,  c_pres, 'ko', e_layer_sizes,  e_recs, 'w^', c_layer_sizes,  c_recs, 'k^', e_layer_sizes, e_scores_f1, 'b.', c_layer_sizes,  c_scores_f1, 'r.')
    (e_lpre, c_lpre) = plt.plot(e_layer_sizes, e_pres, 'ro', c_layer_sizes,  c_pres, 'bo')
    plt.setp(e_lpre, label='Elastic')
    plt.setp(c_lpre, label='Compound')
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend(loc='lower right', numpoints=1)

    plt.subplot(3,1,2)
    (e_lrec, c_lrec) = plt.plot(e_layer_sizes,  e_recs, 'r^', c_layer_sizes,  c_recs, 'b^')
    plt.setp(e_lrec, label='Elastic')
    plt.setp(c_lrec, label='Compound')
    plt.ylabel("Recall")
    plt.grid(True)
    plt.legend(loc='lower right', numpoints=1)

    plt.subplot(3,1,3)
    e_rects = plt.bar(np.asarray(e_layer_sizes), e_scores_f1, 3, alpha=0.8, color='r', error_kw={'ecolor': '0.3'}, label='Elastic')
    c_rects = plt.bar(np.asarray(c_layer_sizes) + 3, c_scores_f1, 3, alpha=0.8, color='b', error_kw={'ecolor': '0.3'}, label='Compound')

    plt.ylabel("F1")
    plt.grid(True)
    plt.legend(loc='lower right', numpoints=1)
    plt.xlabel('Layer Size')

    plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "%s-scores.pdf" % dataset))
    # plt.axis([0, 100, 0, 1])

    # plt.ylabel('Mean Classification Scores')

def plotTimings(figIdx, dataset, e_layer_sizes, e_times, e_times_all, e_epochs, c_layer_sizes, c_times, c_epochs):
    plt.figure(figIdx)
    plt.subplot(3,1,1)
    plt.title("%s - Timing" % dataset)
    # (e_lpre, c_lpre, e_lrec, c_lrec, e_lf1, c_lf1) = plt.plot(e_layer_sizes, e_pres, 'wo', c_layer_sizes,  c_pres, 'ko', e_layer_sizes,  e_recs, 'w^', c_layer_sizes,  c_recs, 'k^', e_layer_sizes, e_scores_f1, 'b.', c_layer_sizes,  c_scores_f1, 'r.')
    (e_ltimes, c_ltimes) = plt.plot(e_layer_sizes, e_times, 'r', c_layer_sizes,  c_times, 'b')
    plt.setp(e_ltimes, label='Elastic')
    plt.setp(c_ltimes, label='Compound')
    plt.ylabel("Training Time [s]")
    plt.grid(True)
    plt.legend(loc='upper left', numpoints=1)

    plt.subplot(3,1,2)
    (e_lepochs, c_lepochs) = plt.plot(e_layer_sizes, e_epochs, 'r', c_layer_sizes,  c_epochs, 'b')
    plt.setp(e_lepochs, label='Elastic')
    plt.setp(c_lepochs, label='Compound')
    plt.ylabel("Epochs")
    plt.grid(True)
    plt.legend(loc='upper left', numpoints=1)
    plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "%s-timing.pdf" % dataset))

    plt.subplot(3,1,3)
    (e_ltimepe, c_ltimepe) = plt.plot(e_layer_sizes, [np.mean(x) for x in e_times_all], 'r', c_layer_sizes, np.asarray(c_times)/np.asarray(c_epochs), 'b')
    plt.setp(e_ltimepe, label='Elastic')
    plt.setp(c_ltimepe, label='Compound')
    plt.ylabel("Time per epoch [s]")
    plt.grid(True)
    plt.legend(loc='upper left', numpoints=1)
    plt.xlabel('Layer Size')
    plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "%s-timing.pdf" % dataset))

def plotErrors(figIdx, dataset, e_layer_sizes, e_errors, c_layer_sizes, c_errors):
    plt.figure(figIdx)
    plt.title("%s - Error" % dataset)
    # (e_lpre, c_lpre, e_lrec, c_lrec, e_lf1, c_lf1) = plt.plot(e_layer_sizes, e_pres, 'wo', c_layer_sizes,  c_pres, 'ko', e_layer_sizes,  e_recs, 'w^', c_layer_sizes,  c_recs, 'k^', e_layer_sizes, e_scores_f1, 'b.', c_layer_sizes,  c_scores_f1, 'r.')
    (e_lerr, c_lerr) = plt.plot(e_layer_sizes, e_errors, 'r', c_layer_sizes,  c_errors, 'b')
    plt.setp(e_lerr, label='Elastic')
    plt.setp(c_lerr, label='Compound')
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend(loc='upper right', numpoints=1)
    plt.xlabel('Layer Size')
    plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "%s-error.pdf" % dataset))

def plotErrorsSep(figIdx, dataset, e_errors_sep, c_errors_sep):
    plt.figure(figIdx)
    plt.title("%s - Error (per layer size)" % dataset)
    for idx, errors in enumerate(e_errors_sep):
        ax = plt.subplot(3,10,idx)
        plt.plot(e_errors_sep[idx], 'r', c_errors_sep[idx], 'b')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.grid(True)
        plt.yticks()
        plt.tight_layout()
    plt.savefig(utils.get_full_path(Config.PATH_EVAL_ROOT, "%s-error-sep.pdf" % dataset))

if __name__ == "__main__":

    dataset = "MNIST"
    maxSize = 301

    fileName = utils.get_full_path(Config.PATH_EVAL_ROOT, dataset, "elastic", "results-elastic-"+dataset+"-"+str(maxSize)+".pkl")
    obj = pickleFromWindowsEncoding(fileName)
    (layer_sizes, scores_f1, pres, recs, errors_sep, times, times_integrated, times_all, times_overall) = extractInfo(obj)

    c_fileName = utils.get_full_path(Config.PATH_EVAL_ROOT, dataset, "compound", "results-compound-"+dataset+"-"+str(maxSize)+".pkl")
    c_obj = pickleFromWindowsEncoding(c_fileName)
    (c_layer_sizes, c_scores_f1, c_pres, c_recs, c_errors_sep, c_times, c_times_integrated, c_times_all, c_times_overall) = extractInfo(c_obj)

    epochs = [len(x) for x in times_all]
    errors = [x[-1] for x in errors_sep]
    c_epochs = [len(x) for x in c_times_all]
    c_errors = [x[-1] for x in c_errors_sep]

    print layer_sizes
    print scores_f1
    print pres
    print recs
    print errors_sep
    print epochs
    print times
    print times_integrated
    print times_all

    print ""
    print c_layer_sizes
    print c_scores_f1
    print c_pres
    print c_recs
    print c_errors_sep
    print c_epochs
    print c_times
    print c_times_integrated
    print c_times_all

    # plt.figure(1)
    # plt.subplot(1,1,1)
    # plt.plot(layer_sizes, scores_f1, 'r', c_layer_sizes, c_scores_f1, 'b')
    # plt.grid(True)
    # plt.xlabel("Hidden Layer Size")
    # plt.ylabel("F1")
    #
    # plt.figure(2)
    # plt.subplot(1,1,1)
    # plt.plot(layer_sizes, times_integrated, 'r', c_layer_sizes, c_times, 'b')
    # plt.grid(True)
    # plt.xlabel("Hidden Layer Size")
    # plt.ylabel("Time[s]")

    plotScores(3, dataset, layer_sizes, pres, recs, scores_f1, c_layer_sizes, c_pres, c_recs, c_scores_f1)
    plotTimings(4, dataset, layer_sizes, times_integrated, times_all, epochs, c_layer_sizes, c_times, c_epochs)
    plotErrors(5, dataset, layer_sizes, errors, c_layer_sizes, c_errors)
    plotErrorsSep(6, dataset, errors_sep, c_errors_sep)
    # plt.figure(6)
    # plt.plot(errors_sep[-1], 'r', c_errors_sep[-1], 'b')
    plt.show()

    # plt.figure(4)
    # plt.subplot(1,1,1)
    # plt.plot(layer_sizes, times_integrated)
    # plt.grid(True)
    # plt.title("Elastic")
    # plt.xlabel('Hidden Layer Size')
    # plt.ylabel('Training Time [s] (per node)')
    # plt.show()
    # c_logs = cPickle.load(open(utils.get_full_path(Config.PATH_EVAL_ROOT, "results-compound.pkl"), 'r'))