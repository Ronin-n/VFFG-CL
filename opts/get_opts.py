import argparse
import os
import torch
import models
import data
import json


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str,
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')
        parser.add_argument('--shared_dir', type=str, default='./shared', help='shared are saved here')
        parser.add_argument('--cuda_benchmark', action='store_true', help='use torch cudnn benchmark')
        parser.add_argument('--has_test', action='store_true',
                            help='whether have test. for 10 fold there is test setting, but in 5 fold there is no test')

        # model parameters
        parser.add_argument('--model', type=str, default='mmin',
                            help='chooses which model to use. [autoencoder | siamese | emotion_A]')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay when training')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012,
                            help='scaling factor for normal, xavier and orthogonal.')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='multimodal',
                            help='chooses how datasets are loaded. [iemocap, ami, mix]')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        ## training parameter
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=80,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        # Mode
        parser.add_argument('--mode', type=str, default='train')  # 模式，默认为训练模式
        parser.add_argument('--runs', type=int, default=5)  # 轮数，默认为5轮

        # Bert
        parser.add_argument('--use_bert', type=str2bool, default=True)  # 是否使用bert，默认为使用
        parser.add_argument('--use_cmd_sim', type=str2bool, default=True)  # 是否使用cmd_sim，默认为使用

        # Train
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--eval_batch_size', type=int, default=10)
        parser.add_argument('--n_epoch', type=int, default=500)
        parser.add_argument('--patience', type=int, default=6)

        parser.add_argument('--diff_weight', type=float, default=0.15)  # default:0.3  3 0.5 0.5 15
        parser.add_argument('--sim_weight', type=float, default=0.025)  # default:1
        parser.add_argument('--sp_weight', type=float, default=0.0)  #
        parser.add_argument('--recon_weight', type=float, default=0.025)  # default:1
        parser.add_argument('--cls_weight', type=float, default=1)  # default:1

        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--clip', type=float, default=1.0)  #

        parser.add_argument('--rnncell', type=str, default='lstm')
        parser.add_argument('--embedding_size', type=int, default=300)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--reverse_grad_weight', type=float, default=1.0)  #
        # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
        parser.add_argument('--activation', type=str, default='relu')

        parser.add_argument('--random_seed', default=336, type=int, help='# the random seed')

        parser.add_argument('--curriculum_stg', type=str, default='single',
                            help='Specify the curriculum stage: single, multiple, mix')
        parser.add_argument('--Is_part_miss', type=str2bool, default=False)  # 是否部分缺失
        parser.add_argument('--miss_rate', type=float, default=0.1)

        parser.add_argument('--early_stage_epochs', default=10, type=int, help='# the random seed')


        # ---------END---------

        # expr setting 
        parser.add_argument('--run_idx', type=str, help='experiment number; for repeat experiment')
        self.isTrain = True
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        print(f"model name : {model_name}")
        curriculum_stg = opt.curriculum_stg

        if 'CL' in model_name:
            print(f"curriculum learning strategy : {curriculum_stg}")
            model_option_setter = models.get_option_setter(model_name, curriculum_stg)
        else:
            model_option_setter = models.get_option_setter(model_name, curriculum_stg)

        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)  # 这个函数目前没东西，parser还是原来的parser

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        if opt.verbose:
            print(message)

        # save to the disk
        model_name = opt.model
        if 'CL' in model_name:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.curriculum_stg)
        else:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        # log dir
        log_dir = os.path.join(opt.log_dir, opt.name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # save opt as txt file
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def save_json(self, opt):
        dictionary = {}
        for k, v in sorted(vars(opt).items()):
            dictionary[k] = v
        model_name = opt.model
        if 'CL' in model_name:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.curriculum_stg)
        else:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        save_path = os.path.join(expr_dir, '{}_opt.conf'.format(opt.phase))
        json.dump(dictionary, open(save_path, 'w'), indent=4)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
            print("Expr Name:", opt.name)

        self.print_options(opt)

        if opt.isTrain:
            self.save_json(opt)

        self.opt = opt
        return self.opt
