import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pudb


class UnetCNNModel(BaseModel):
    def name(self):
        return 'UnetCNNModel'

    def initialize(self, opt):
        ''' Parses opts and initializes the relevant networks. '''
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load and define networks according to opts
        self.netG = networks.define_G(
                opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = False
            # use_sigmoid = opt.no_lsgan
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            # if self.isTrain:
            #     self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            # self.criterionGAN = networks.GANLoss(
            #        use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                    self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(
            #        self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized ----------')
        networks.print_network(self.netG)
        # if self.isTrain:
        #     networks.print_network(self.netD)
        print('------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # run this after setting inputs
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        # no backprop on gradients
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        self.loss_G_L2 = self.criterionL2(self.fake_B,
                self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_L2

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # self.optimizer_D.zero_grad()
        # self.backward_D()
        # self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_L2', self.loss_G_L2.data[0]), ('Test', self.loss_G_L2.data[0])])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
