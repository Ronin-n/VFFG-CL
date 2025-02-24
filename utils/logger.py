import time
import os
import logging
import fcntl

def get_logger(path, suffix):
    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S',time.localtime(time.time()))
    logger = logging.getLogger(__name__+cur_time)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(path, f"{suffix}_{cur_time}.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class ResultRecorder(object):
    def __init__(self, path, total_cv, corpus_name):
        self.path = path
        self.total_cv = total_cv
        self.corpus_name = corpus_name
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                if corpus_name != 'MOSI':
                    f.write('acc\tuar\tmacro_f1\tweighted_f1\n')
                else:
                    f.write('acc\tMAE\tcorr\tf1\n')

    def is_full(self, content):
        # Checking if content has all results and each line has 4 valid entries
        if len(content) < self.total_cv + 1:
            return False

        for line in content[1:]:  # Skip header
            if len(line.split()) != 4:  # Expecting 4 columns of data
                return False
        return True

    def calc_mean(self, content, corpus_name):
        if corpus_name != 'MOSI':
            # Extracting data from each line
            acc = [float(line.split()[0]) for line in content[1:self.total_cv + 1]]
            uar = [float(line.split()[1]) for line in content[1:self.total_cv + 1]]
            macro_f1 = [float(line.split()[2]) for line in content[1:self.total_cv + 1]]
            weighted_f1 = [float(line.split()[3]) for line in content[1:self.total_cv + 1]]

            mean_acc = sum(acc) / len(acc)
            mean_uar = sum(uar) / len(uar)
            mean_macro_f1 = sum(macro_f1) / len(macro_f1)
            mean_weighted_f1 = sum(weighted_f1) / len(weighted_f1)

            return mean_acc,mean_uar, mean_macro_f1, mean_weighted_f1
        else:
            acc = [float(line.split()[0]) for line in content[1:self.total_cv + 1]]
            mae = [float(line.split()[1]) for line in content[1:self.total_cv + 1]]
            corr = [float(line.split()[2]) for line in content[1:self.total_cv + 1]]
            f1 = [float(line.split()[3]) for line in content[1:self.total_cv + 1]]

            # Calculating means
            mean_acc = sum(acc) / len(acc)
            mean_mae = sum(mae) / len(mae)
            mean_corr = sum(corr) / len(corr)
            mean_f1 = sum(f1) / len(f1)

            return mean_acc, mean_mae, mean_corr, mean_f1


    def write_result_to_tsv(self, results, cvNo, corpus_name):
        # Use fcntl for file locking to avoid race conditions
        with open(self.path) as f_in:
            fcntl.flock(f_in.fileno(), fcntl.LOCK_EX)
            content = f_in.readlines()

        # Extend the content if necessary
        if len(content) < self.total_cv + 1:
            content += ['\n'] * (self.total_cv - len(content) + 1)

        # Writing the current result
        if corpus_name != 'MOSI':
            content[cvNo] = '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
             results['acc'], results['uar'], results['macro_f1'], results['weighted_f1']
            )
        else:
            content[cvNo] = '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
             results['acc'], results['mae'], results['corr'], results['f1']
            )

        # If all results are available, calculate and append the mean
        if self.is_full(content):
            if corpus_name != 'MOSI':
                mean_acc, mean_uar, mean_macro_f1, mean_weighted_f1 = self.calc_mean(content, corpus_name)
                content.append('Mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                    mean_acc, mean_uar, mean_macro_f1, mean_weighted_f1
                ))
            else:
                mean_acc, mean_mae, mean_corr, mean_f1 = self.calc_mean(content, corpus_name)
                content.append('Mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                    mean_acc, mean_mae, mean_corr, mean_f1
                ))

        # Writing back the content to the file
        with open(self.path, 'w') as f_out:
            f_out.writelines(content)


class LossRecorder(object):
    def __init__(self, path, total_cv=10, total_epoch=40):
        self.path = path
        self.total_epoch = total_epoch
        self.total_cv = total_cv
        if not os.path.exists(self.path):
            f = open(self.path, 'w')
            f.close()

    def is_full(self, content):
        if len(content) < self.total_cv + 1:
            return False

        for line in content:
            if not len(line.split('\t')) == 3:
                return False
        return True

    def calc_mean(self, content):
        loss_list = [[] * self.total_cv] * self.total_epoch
        mean_list = [[] * self.total_cv] * self.total_epoch
        for i in range(0, self.total_epoch):
            loss_list[i] = [float(line.split('\t')[i]) for line in content[1:]]
        for i in range(0, self.total_epoch):
            mean_list[i] = sum(loss_list[i]) / len(loss_list[i])
        return mean_list

    def write_result_to_tsv(self, results, cvNo):
        # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
        f_in = open(self.path)
        fcntl.flock(f_in.fileno(), fcntl.LOCK_EX)  # 加锁
        content = f_in.readlines()
        if len(content) < self.total_cv + 1:
            content += ['\n'] * (self.total_cv - len(content) + 1)
        string = ''
        for i in results:
            string += str(i.numpy())[:8]
            string += '\t'
        content[cvNo] = string + '\n'

        f_out = open(self.path, 'w')
        f_out.writelines(content)
        f_out.close()
        f_in.close()  # 释放锁

    def read_result_from_tsv(self,):
        f_out = open(self.path)
        fcntl.flock(f_out.fileno(), fcntl.LOCK_EX)
        content = f_out.readlines()
        loss_list = [[] * self.total_cv] * self.total_epoch
        for i in range(0, self.total_epoch):
            loss_list[i] = [float(line.split('\t')[i]) for line in content[1:]]
        mean = self.calc_mean(content)
        return mean
