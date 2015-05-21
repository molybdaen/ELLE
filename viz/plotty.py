__author__ = 'johannesjurgovsky'

import cPickle
import matplotlib.pyplot as plt
from utils import utils
from config import Config
import numpy as np
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