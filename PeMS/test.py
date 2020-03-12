import math
import argparse
import utils
import time
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 12,
                    help = 'prediction steps')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'batch size')
parser.add_argument('--traffic_file', default = 'data/PeMS.h5',
                    help = 'traffic file')
parser.add_argument('--SE_file', default = 'data/SE(PeMS).txt',
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = 'data/GMAN(PeMS)',
                    help = 'pre-trained model')
parser.add_argument('--log_file', default = 'data/log(PeMS)',
                    help = 'log file')
args = parser.parse_args()

start = time.time()

log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
 SE, mean, std) = utils.loadData(args)
num_train, num_val, num_test = trainX.shape[0], valX.shape[0], testX.shape[0]
utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
graph = tf.Graph()
with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph = graph, config = config) as sess:
    saver.restore(sess, args.model_file)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    pred = graph.get_collection(name = 'pred')[0]
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    trainPred = []
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            'Placeholder:0': trainX[start_idx : end_idx],
            'Placeholder_1:0': trainTE[start_idx : end_idx],
            'Placeholder_3:0': False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        trainPred.append(pred_batch)
    trainPred = np.concatenate(trainPred, axis = 0)
    valPred = []
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            'Placeholder:0': valX[start_idx : end_idx],
            'Placeholder_1:0': valTE[start_idx : end_idx],
            'Placeholder_3:0': False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        valPred.append(pred_batch)
    valPred = np.concatenate(valPred, axis = 0)
    testPred = []
    num_batch = math.ceil(num_test / args.batch_size)
    start_test = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            'Placeholder:0': testX[start_idx : end_idx],
            'Placeholder_1:0': testTE[start_idx : end_idx],
            'Placeholder_3:0': False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        testPred.append(pred_batch)
    end_test = time.time()
    testPred = np.concatenate(testPred, axis = 0)
train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
test_mae, test_rmse, test_mape = utils.metric(testPred, testY)
utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                 (train_mae, train_rmse, train_mape * 100))
utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae, val_rmse, val_mape * 100))
utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae, test_rmse, test_mape * 100))
utils.log_string(log, 'performance in each prediction step')
MAE, RMSE, MAPE = [], [], []
for q in range(args.Q):
    mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                     (q + 1, mae, rmse, mape * 100))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)
utils.log_string(
    log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
    (average_mae, average_rmse, average_mape * 100))
end = time.time()
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
