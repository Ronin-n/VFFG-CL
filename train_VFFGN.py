import os
import time
import numpy as np
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder, LossRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from utils.early_stopping import EarlyStopping
import random
import pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_part(part_names, total_miss_type, total_pred, total_label, save_dir, phase, recorder_lookup, cvNo):
    for part_name in part_names:
        part_index = np.where(total_miss_type == part_name)
        part_pred = total_pred[part_index]
        part_label = total_label[part_index]
        non_zeros = np.array([i for i, e in enumerate(part_label) if e != 0])
        acc_part = accuracy_score(part_label, part_pred)
        uar_part = recall_score(part_label, part_pred, average='macro')
        macro_f1_part = f1_score(part_label, part_pred, average='macro')
        f1_part = f1_score((part_label[non_zeros] > 0), (part_pred[non_zeros] > 0), average='macro')
        weighted_f1_part = f1_score(part_label, part_pred, average='weighted')
        mae_part = np.mean(np.absolute(part_pred - part_label))
        corr_part = np.corrcoef(part_pred, part_label)[0][1]

        np.save(os.path.join(save_dir, f'{phase}_{part_name}_pred.npy'), part_pred)
        np.save(os.path.join(save_dir, f'{phase}_{part_name}_label.npy'), part_label)

        if phase == 'test':
            recorder_lookup[part_name].write_result_to_tsv({
                'acc': acc_part,
                'uar': uar_part,
                'macro_f1': macro_f1_part,
                'weighted_f1': weighted_f1_part
            }, cvNo=cvNo, corpus_name=opt.corpus_name)


def eval(model, val_iter, is_save=False, phase='test', epoch=-1, mode=None):
    model.eval()

    total_pred = []
    total_label = []
    total_miss_type = []
    total_data = 0

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        total_data += 1
        model.set_input(data)
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        miss_type = np.array(data['miss_type'])

        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)

    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    macro_f1 = f1_score(total_label, total_pred, average='macro')
    weighted_f1 = f1_score(total_label, total_pred, average='weighted')
    # cm = confusion_matrix(total_label, total_pred)

    if is_save:
        # save test whole results
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

        # save part results
        if 'CL' in opt.model:
            if opt.curriculum_stg == 'single':
                process_part(['avz', 'azl', 'zvl'], total_miss_type, total_pred, total_label, save_dir, phase,
                             recorder_lookup, opt.cvNo)
            elif opt.curriculum_stg == 'multiple':
                process_part(['azz', 'zvz', 'zzl'], total_miss_type, total_pred, total_label, save_dir, phase,
                             recorder_lookup, opt.cvNo)
            else:
                process_part(['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'], total_miss_type, total_pred, total_label,
                             save_dir, phase, recorder_lookup, opt.cvNo)

        model.train()

        return acc, uar, macro_f1, weighted_f1  # , cm

    else:
        acc, mae, corr, f1 = calc_metrics(total_label, total_pred, mode)

        # save test results
        if is_save:
            save_dir = model.save_dir
            np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
            np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

            if 'CL' in opt.model:
                if opt.curriculum_stg == 'single':
                    process_part(['avz', 'azl', 'zvl'], total_miss_type, total_pred, total_label, save_dir, phase,
                                 recorder_lookup, opt.cvNo)
                elif opt.curriculum_stg == 'multiple':
                    process_part(['azz', 'zvz', 'zzl'], total_miss_type, total_pred, total_label, save_dir, phase,
                                 recorder_lookup, opt.cvNo)
                else:
                    process_part(['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'], total_miss_type, total_pred, total_label,
                                 save_dir, phase, recorder_lookup, opt.cvNo)
            else:
                process_part(['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'], total_miss_type, total_pred, total_label,
                             save_dir, phase, recorder_lookup, opt.cvNo)

        model.train()

        return acc, mae, corr, f1


def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(random_seed)


def calc_metrics(y_true, y_pred, mode=None, to_print=False):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """

    test_preds = y_pred.squeeze(1)
    test_truth = y_true
    # non-neg - neg
    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    acc = accuracy_score(binary_truth, binary_preds)
    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    macro_f1 = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='macro')
    weighted_f1 = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

    return acc, mae, corr, macro_f1


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def pad_to_same_shape(features, target_dim):
    """
    将输入特征补零到指定的列数。
    Args:
        features (numpy.ndarray): 输入特征，形状 (N, D)
        target_dim (int): 目标列数
    Returns:
        numpy.ndarray: 补齐后的特征
    """
    padded_features = np.zeros((features.shape[0], target_dim))
    padded_features[:, :features.shape[1]] = features
    return padded_features


if __name__ == '__main__':
    opt = Options().parse()  # get training options
    opt.gpu_ids = [int(id) for id in opt.gpu_ids.split(',') if id.strip().isdigit()]
    # set_random_seed(opt.random_seed)    # Setting random seed
    if 'CL' in opt.model:
        logger_path = os.path.join(opt.log_dir, opt.name, opt.curriculum_stg, str(opt.cvNo))  # get logger path
    else:
        logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))
    if not os.path.exists(logger_path):  # make sure logger path exists
        os.makedirs(logger_path)
    print(f"logger save path : {logger_path}")

    if 'CL' in opt.model:
        result_dir = os.path.join(opt.log_dir, opt.name, opt.curriculum_stg, 'results')
    else:
        result_dir = os.path.join(opt.log_dir, opt.name, 'results')

    if not os.path.exists(result_dir):  # make sure result path exists
        os.makedirs(result_dir)
    total_cv = 10 if opt.corpus_name != 'MSP' else 12
    if 'CL' in opt.model:
        if opt.curriculum_stg == 'single':
            recorder_lookup = {  # init result recoreder
                "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "avl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv,
                                      corpus_name=opt.corpus_name)
            }
        elif opt.curriculum_stg == 'multiple':
            recorder_lookup = {  # init result recoreder
                "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "avl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv,
                                      corpus_name=opt.corpus_name)
            }
        else:
            recorder_lookup = {  # init result recoreder
                "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv, corpus_name=opt.corpus_name),
                "avl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv,
                                      corpus_name=opt.corpus_name)
            }
    else:
        recorder_lookup = {  # init result recoreder
            "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv,
                                    corpus_name=opt.corpus_name),
            "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'), total_cv=total_cv,
                                  corpus_name=opt.corpus_name),
            "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'), total_cv=total_cv,
                                  corpus_name=opt.corpus_name),
            "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'), total_cv=total_cv,
                                  corpus_name=opt.corpus_name),
            "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'), total_cv=total_cv,
                                  corpus_name=opt.corpus_name),
            "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'), total_cv=total_cv,
                                  corpus_name=opt.corpus_name),
            "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv,
                                  corpus_name=opt.corpus_name),
            "avl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv,
                                  corpus_name=opt.corpus_name)
        }
    loss_dir = os.path.join(opt.image_dir, opt.name, 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    recorder_loss = LossRecorder(os.path.join(loss_dir, 'result_loss.tsv'), total_cv=total_cv,
                                 total_epoch=opt.niter + opt.niter_decay)

    suffix = '_'.join([opt.model, opt.dataset_mode])  # get logger suffix
    logger = get_logger(logger_path, suffix)  # get logger

    if opt.has_test:  # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
    dataset_size = len(dataset)  # get the number of images in the dataset.
    tst_dataset_size = len(tst_dataset)
    logger.info('The number of training samples = %d' % dataset_size)
    logger.info('The number of testing samples = %d' % tst_dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    best_eval_epoch = -1  # record the best eval epoch
    best_eval_acc, best_eval_uar, best_eval_macro_f1, best_eval_weighted_f1, best_eval_f1, best_eval_corr, best_eval_mae = 0, 0, 0, 0, 0, 0, 10

    # # initing
    # early_stopping = EarlyStopping(model.save_dir)

    for epoch in range(opt.epoch_count,
                   opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        loss_add = True

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1  # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights

            # if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #     logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
            #                 ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
            iter_data_time = time.time()

        losses = model.get_current_losses()
        logger.info('\nCur epoch {}'.format(epoch) + ' loss ' +
                    ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)  # update learning rates at the end of every epoch.

        # eval
        acc, uar, macro_f1, weighted_f1 = eval(model, val_dataset, epoch)
                logger.info('Val result of epoch %d / %d acc %.4f uar %.4f macro_f1 %.4f weighted_f1 %.4f' % (
                    epoch, opt.niter + opt.niter_decay, acc, uar, macro_f1, weighted_f1))

        # logger.info('\n{}'.format(cm))

        # show test result for debugging
        if opt.has_test and opt.verbose:
            acc, uar, macro_f1, weighted_f1 = eval(model, tst_dataset, epoch)
                    logger.info(
                        'Tst result of epoch %d / %d acc %.4f uar %.4f macro_f1 %.4f weighted_f1 %.4f' % (
                            epoch, opt.niter + opt.niter_decay, acc, uar, macro_f1, weighted_f1))
                    
            # logger.info('\n{}'.format(cm))

        # record epoch with best result
        if opt.corpus_name == 'IEMOCAP' or 'MSP':
            if uar > best_eval_uar:
                best_eval_epoch = epoch
                best_eval_uar = uar
                best_eval_acc = acc
                best_eval_macro_f1 = macro_f1
                best_eval_weighted_f1 = weighted_f1
            select_metric = 'uar'
            best_metric = best_eval_uar
        else:
            raise ValueError(f'corpus name must be IEMOCAP or MSP, but got {opt.corpus_name}')

    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))
    # test
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        _ = eval(model, val_dataset, is_save=True, phase='val', epoch=best_eval_epoch)
        acc, uar, macro_f1, weighted_f1 = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
        logger.info('Tst result acc %.4f uar %.4f macro_f1 %.4f weighted_f1 %.4f' % (acc, uar, macro_f1, weighted_f1))
        # logger.info('\n{}'.format(cm))
        recorder_lookup['total'].write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }, cvNo=opt.cvNo, corpus_name=opt.corpus_name)
    else:
        recorder_lookup['total'].write_result_to_tsv({
            'acc': best_eval_acc,
            'uar': best_eval_uar,
            'macro_f1': best_eval_macro_f1,
            'weighted_f1': best_eval_weighted_f1
        }, cvNo=opt.cvNo, corpus_name=opt.corpus_name)
    if 'CL' in opt.model:
        clean_chekpoints(opt.name + '/' + opt.curriculum_stg + '/' + str(opt.cvNo), best_eval_epoch)
    else:
        clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
