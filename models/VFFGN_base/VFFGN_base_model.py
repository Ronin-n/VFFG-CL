import torch
import os
import json
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier, Fusion
from models.networks.autoencoder_2 import ResidualAE, MultimodalFusion
from models.utils.config import OptConfig
from models.RFFP_model import RFFPModel
from einops import rearrange
import itertools
from torch import nn


def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)


def log(t, eps=1e-20):
    return torch.log(t + eps)


def l2norm(t):
    return F.normalize(t, dim=-1)


class VFFGNCLbaseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--pretrained_consistent_path', type=str,
                            help='where to load pretrained consistent encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cl_weight', type=float, default=1.0, help='weight of cl loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--ist_weight', type=float, default=1.0, help='weight of ist loss')
        parser.add_argument('--share_weight', action='store_true',
                            help='share weight of forward and backward autoencoders')
        parser.add_argument('--image_dir', type=str, default='./consistent_image', help='models image are saved here')

        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE', 'CL', 'IST']
        self.model_names = ['C', 'AE', 'Fusion']

        # acoustic model
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a).to(self.device)
        self.model_names.append('A')

        # lexical model 
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l, dropout=0.5).to(self.device)
        self.model_names.append('L')

        # visual model
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v).to(self.device)
        self.model_names.append('V')

        # Residual Auto-encoder
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = opt.embd_size_a + \
                       opt.embd_size_v + \
                       opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False).to(self.device)
        
        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a + \
                         opt.embd_size_v + \
                         opt.embd_size_l
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                     use_bn=opt.bn).to(self.device)
        # Multimodal fusion module
        self.netFusion = MultimodalFusion(input_dim=cls_input_size, kernel_size=3).to(self.device)

        self.temperature = torch.nn.Parameter(torch.tensor(1.))   

        if self.isTrain:
            self.load_pretrained_encoder(opt)
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cl_weight = opt.cl_weight
            self.ist_weight = opt.ist_weight
            self.cycle_weight = opt.cycle_weight
            # L2 regularization
            self.l2_lambda = 1e-5
            self.l2_params = list(
                itertools.chain(*[getattr(self, 'net' + model).parameters() for model in self.model_names]))
        else:
            self.load_pretrained_encoder(opt)

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, opt.curriculum_stg, str(opt.cvNo))
        print(f"checkpoints save path : {self.save_dir}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


        image_save_dir = os.path.join(opt.image_dir, opt.name)
        image_save_dir = os.path.join(image_save_dir, str(opt.cvNo))
        self.predict_image_save_dir = os.path.join(image_save_dir, 'predict')
        self.consistent_image_save_dir = os.path.join(image_save_dir, 'consistent')
        self.loss_image_save_dir = os.path.join(image_save_dir, 'loss')
        if not os.path.exists(self.predict_image_save_dir):
            os.makedirs(self.predict_image_save_dir)
        if not os.path.exists(self.consistent_image_save_dir):
            os.makedirs(self.consistent_image_save_dir)
        if not os.path.exists(self.loss_image_save_dir):
            os.makedirs(self.loss_image_save_dir)

    # Load Pre-trained Encoder
    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False  # teacher model should be in test mode
        pretrained_config.gpu_ids = opt.gpu_ids  # set gpu to the same
        self.pretrained_encoder = RFFPModel(pretrained_config)
        self.pretrained_encoder.load_networks_cv(pretrained_path)
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()


    # Initialize Encoder
    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))
            self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))
            self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))

            self.netFusion.load_state_dict(f(self.pretrained_encoder.netFusion.state_dict()))

            self.netC.load_state_dict(f(self.pretrained_encoder.netC.state_dict()))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    # load dataset
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.acoustic = acoustic = input['A_feat'].float().to(self.device)
        self.lexical = lexical = input['L_feat'].float().to(self.device)
        self.visual = visual = input['V_feat'].float().to(self.device)

        if self.isTrain:
            self.label = input['label'].to(self.device)
            self.missing_index = input['missing_index'].long().to(self.device)  # [a,v,l]

            # A modality
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
            self.A_miss = acoustic * self.A_miss_index
            self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
            # V modality
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
            self.V_miss = visual * self.V_miss_index
            self.V_reverse = visual * -1 * (self.V_miss_index - 1)
            # L modality
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
            self.L_miss = lexical * self.L_miss_index
            self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
            if self.opt.corpus_name == 'MOSI':
                self.label = self.label.unsqueeze(1)
        else:
            self.A_miss = acoustic
            self.V_miss = visual
            self.L_miss = lexical


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.feat_A_miss = self.netA(self.A_miss).to(self.device)  # missing modaltity feature
        self.feat_V_miss = self.netV(self.V_miss).to(self.device)  # missing modaltity feature
        self.feat_L_miss = self.netL(self.L_miss) .to(self.device)  # missing modaltity feature

        # fusion miss
        self.feat_fusion = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1).to(self.device)

        self.feat_VFF, _ = self.netAE(self.feat_fusion)

        # get fusion outputs for missing modality
        self.logits, _ = self.netC(self.feat_VFF)
        # self.logits, _ = self.pretrained_encoder.netC(self.recon_fusion)

        if self.opt.corpus_name != 'MOSI':
            self.pred = F.softmax(self.logits, dim=-1)
        else:
            self.pred = self.logits

        # for training
        if self.isTrain:
            with torch.no_grad():
                self.embd_A_real = self.pretrained_encoder.netA(self.acoustic)
                # print(f"embd_A_consistent  shape: {embd_A_consistent.shape}")
                self.embd_L_real = self.pretrained_encoder.netL(self.lexical)
                self.embd_V_real = self.pretrained_encoder.netV(self.visual)
                self.feat_real = torch.cat([self.embd_A_real, self.embd_L_real, self.embd_V_real], dim=-1)

                self.feat_RFF = self.pretrained_encoder.netFusion(self.feat_real)

    def backward(self):
        """Comparative learning"""
        temp = self.temperature.exp()

        # Calculate similarity matrix st
        x_virtual = F.normalize(self.feat_VFF, p=2, dim=1)
        x_original = F.normalize(self.feat_RFF, p=2, dim=1)
        st = torch.mm(x_original, x_virtual.t())

        # Normalize similarity matrix
        yt_to_vt = F.log_softmax(st / temp, dim=1)
        yvt_to_t = F.log_softmax(st.t() / temp, dim=1)


        # Ground truth one-hot similarity
        N = x_original.size(0)
        y_true = torch.eye(N, device=x_original.device)

        # Cross-entropy loss with logits
        loss_t_to_vt = F.kl_div(yt_to_vt, y_true, reduction='batchmean')
        loss_vt_to_t = F.kl_div(yvt_to_t, y_true, reduction='batchmean')

        # Similarity alignment loss Lsa
        self.loss_CL = self.cl_weight * 0.5*(loss_t_to_vt + loss_vt_to_t)
        # print(f"loss_cl: {self.loss_cl}")

        # L2 regularization
        l2_reg = sum(param.pow(2.0).sum() for param in self.l2_params)

        # Instruction loss
        self.loss_IST = self.ist_weight * self.criterion_mse(self.feat_real, self.feat_fusion)

        # Classification loss
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)

        # Final loss
        loss = self.loss_CE + self.loss_CL + self.loss_IST
        loss.backward()

        # Clip gradients
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
