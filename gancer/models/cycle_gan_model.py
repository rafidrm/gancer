import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pudb


class CycleGANModel(BaseModel):

    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        ''' Parses opts and initializes the relevant networks. '''
        raise NotImplementedError

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
        pass

    def test(self):
        # no backprop on gradients
        pass

    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        # self.forward()
        pass
