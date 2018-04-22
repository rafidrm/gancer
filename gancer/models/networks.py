import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import pudb

#
# Functions
#


def weights_init_normal(m):
    ''' Initializes m.weight tensors with normal dist'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    ''' Initializes m.weight tensors with normal dist (Xavier algorithm)'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    ''' Initializes m.weight tensors with He algorithm.'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [{}]'.format(init_type))
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{}] is not implemented'.format(init_type))


def get_norm_layer(norm_type='instance'):
    ''' Applies batch norm or instance norm. Batch norm: normalize the output of
    every layer before applying activation function, i.e., normalize the acti-
    vations of the previous layer for each batch. In instance normalization, we
    normalize the activations of the previous layer on a per-image/data point
    basis.
    '''
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'batch_3d':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance_3d':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [{}] is not found'.format(norm_type))
    return norm_layer


def get_scheduler(optimizer, opt):
    ''' Rules for how to adjust the learning rate. Lambda: custom method to
    change learning rate. StepLR: learning rate decays by gamma each step size.
    Plateau: reduce once the quantity monitored has stopped decreasing.
    '''
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - \
                max(0, epoch + 1 + opt.epoch_count - opt.niter) / \
                float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [{}] is not implemented'.format(
                opt.lr_policy))
    return scheduler


def define_G(input_nc,
             output_nc,
             ngf,
             which_model_netG,
             norm='batch',
             use_dropout=False,
             init_type='normal',
             gpu_ids=[]):
    ''' Parses model parameters and defines the Generator module.
    '''
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
            gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=6,
            gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(
            input_nc,
            output_nc,
            7,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(
            input_nc,
            output_nc,
            8,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64_3d':
        netG = UnetGenerator(
            input_nc,
            output_nc,
            6,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            conv=nn.Conv3d,
            deconv=nn.ConvTranspose3d)
    elif which_model_netG == 'unet_128_3d':
        netG = UnetGenerator(
            input_nc,
            output_nc,
            7,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            conv=nn.Conv3d,
            deconv=nn.ConvTranspose3d)
    elif which_model_netG == 'unet_cnn':
        netG = UnetCNNGenerator(
            input_nc,
            output_nc,
            7,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            gpu_ids=gpu_ids)
    else:
        raise NotImplementedError(
            'Generator model name [{}] is not recognized'.format(
                which_model_netG))
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc,
             ndf,
             which_model_netD,
             n_layers_D=3,
             norm='batch',
             use_sigmoid=False,
             init_type='normal',
             gpu_ids=[]):
    ''' Parses model parameters and defines the Discriminator module.
    '''
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers=3,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers_D,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(
            input_nc,
            ndf,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers_3d':
        netD = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers_D,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids,
            conv=nn.Conv3d)
    elif which_model_netD == 'voxel':
        netD = PixelDiscriminator(
            input_nc,
            ndf,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids,
            conv=nn.Conv3d)
    elif which_model_netD == 'gan_3d':
        netG = Gan3dDiscriminator(
            input_nc, ngf, 3, norm_layer=norm_layer, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError(
            'Discriminator model name [{}] is not recognized'.format(
                which_model_netD))

    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {}'.format(num_params))


#
# Classes
#


class GANLoss(nn.Module):
    ''' Defines GAN loss func, which uses either LSGAN or regular GAN. LSGAN is
    effectively just MSELoss, but abstracts away the need to create the target
    label tensor that has the same size as the input.
    '''

    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()  # mean squared error
        else:
            self.loss = nn.BCELoss()  # binary cross entropy

    def get_target_tensor(self, input, target_is_real):
        ''' Loss function needs 2 inputs, an 'input' and a target tensor. If 
        the target is real, then create a 'target tensor' filled with real
        label (1.0) everywhere. If the target is false, then create a 'target
        tensor' filled with false label (0.0) everywhere. Then do BCELoss or
        MSELoss as desired.
        '''
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetGenerator(nn.Module):
    ''' Defines a generator comprised of Resnet blocks between a few down-
    sampling/upsampling operations. Code and idea originally from Justin
    Johnson's architecture. https://github.com/jcjohnson/fast-neural-style/

    Init:
        - Reflection Pad: 3px on all sides
        - Conv2d: input -> 64 channels, 7x7 kernels, no padding
        - Normalize -> ReLU

    Downsample:
        - Conv2d: 64 -> 128 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU
        - Conv2d: 128 -> 256 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU

    Resnet: (x n_blocks)
        - Resnet: 256 -> 256 channels (Conv -> ReLU -> Conv -> ReLU + x)

    Upsample:
        - Deconv2d: 256 -> 128 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU
        - Deconv2d: 128 -> 64 channels, 3x3 kernels, 2 stride, 1 padding
        - Normalize -> ReLU

    Out:
        - Reflection Pad: 3px on all sides
        - Conv2d: 64 -> output channels, 7x7 kernels, no padding
        - Tanh
    '''

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=6,
                 gpu_ids=[],
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # input and output number of channels
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        # bias only if we're doing instance normalization
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # ReflectionPad2D: take each image and pad it with 3px on all sides
        # Conv2d: outs 64 channels with 7x7 kernels
        # norm_layer: 2-D batch norm on all 64 channels
        # RelU: ReLU on
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i  # mult = 1, 2
            # add 128, and 256 channels with 3x3 kernels
            # 2=D bach norm on all 128, then 256 channels
            # ReLU after each channel
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2**n_downsampling  # 4
        for i in range(n_blocks):  # default 6
            # Add resnet block
            model += [
                ResnetBLock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias)
            ]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class ResnetBlock(nn.Module):
    ''' Resnets reduce the vanishing gradient problem. The block is structured:

        x -> Conv2d -> ReLU -> Conv2d -> out + x

    Effectively, the input x is added back at the output (see self.forward()).
    Each Conv2d block is 3x3 kernels with fixed dimension input and output
    channels.
    '''

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_dropout, use_bias)

    def build_conv_block(self,
                         dim,
                         padding_type,
                         norm_layer,
                         use_dropout,
                         use_bias,
                         conv=nn.Conv2d):
        conv_block = []
        p = 0
        # add 1px padding to input
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [{}] is not implemented'.format(padding_type))

        # 2d conv, same number of output channels as input, 3x3 kernels
        # Then batch normalize and RelU, and dropout if needed
        conv_block += [
            conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [{}] is not implemented'.format(padding_type))

        # 2d conv, same number of output channels as input 3x3 kernels, then
        # batch normalize
        conv_block += [
            conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetGenerator(nn.Module):
    ''' Defines the UNet generator architecture proposed in Isola et al.
    Architecture contains downsampling/upsampling operations, with Unet
    connectors.

    - Outermost Unet:
        - Conv2d: InputC -> 64, 4x4 kernels

    - Outer Unets:
        - Leaky ReLU (iv)
        - Conv2d: 64 -> 128, 4x4 kernels
        - Leaky ReLU (iii)
        - Conv2d: 128 -> 256, 4x4 kernels
        - Leaky ReLU (ii)
        - Conv2d: 256 -> 512, 4x4 kernels

    - Intermediate Unets (x num_downs):
        - Leaky ReLU (i)
        - Conv2d: 512 -> 512, 4x4 kernels

    - Inner Unet:
        - Leaky ReLU (a)
        - Conv2d: 512 -> 512, 4x4 kernels
        - ReLU
        - Deconv2d: 512 -> 512, 4x4 kernels
        - Normalize --> Connect to (a)

    - Intermediate Unets continued:
        - ReLU
        - Deconv2d: 1024 -> 512, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (i)

    - Outer Unets:
        - ReLU
        - Deconv2d: 512 -> 256, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (ii)
        - ReLU
        - Deconv2d: 256 -> 128, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iii)
        - ReLU
        - Deconv2d: 128 -> 64, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iv)

    - Outermost Unet:
        - ReLU
        - Deconv2d: 128 -> outC, 4x4 kernels
        - Tanh
    '''

    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 gpu_ids=[],
                 conv=nn.Conv2d,
                 deconv=nn.ConvTranspose2d):
        '''
        Args:
            num_downs:  number of downsamplings in the Unet.
        '''
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # blocks are built recursively starting from innermost moving out
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            conv=conv,
            deconv=deconv)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                conv=conv,
                deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    ''' Unet Skip Connection built recursively by taking a submodule and
    generating a downsample and upsample block over it. These blocks are then
    connected in a short circuit.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 conv=nn.Conv2d,
                 deconv=nn.ConvTranspose2d):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # bias only if we're doing instance normalization
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d) or (
                norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (
                norm_layer == nn.InstanceNorm3d)
        if input_nc is None:
            input_nc = outer_nc
        # basic building blocks
        # Conv2d: inputC -> innerC, 4x4 kernel size, 2 stride, 1 padding
        downconv = conv(
            input_nc,
            inner_nc,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # Conv2d: inputC -> innerC, 4x4 kernel size, 2 stride, 1 padding
            # then submodule
            # ReLU -> Deconv2d: innerC*2 -> outerC, 4x4 kernel size, ...
            # Tanh
            upconv = deconv(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # LeakyReLU -> Conv2d
            # ReLU -> Deconv2d: innerC -> outerC, 4x4 kernel size, 2 stride...
            # Normalize
            upconv = deconv(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # LeakyReLU -> Conv2d -> Normalize
            # then submodule
            # ReLU -> Deconv2d -> Normalize
            upconv = deconv(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Apply connections if inner modules
            # TODO: might need to check if cat dimension needs to change
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    ''' PatchGAN discriminator, supposed to work on patches within the full image
    to evaluate whether the style is transferred everywhere.

    Init:
        - Conv2d: inputC -> ndf, 4x4 kernel size
        - LeakyReLU

    Intermediate:
        - Conv2D: ndf -> 8 * ndf, 4x4 kernel
        - Normalize -> LeakyReLU
        - Conv2d: 8 * ndf -> 8 * ndf, 4x4 kernel
        - Normalize -> LeakyReLU
        - Conv2d: 8 * ndf -> 8 * ndf, 4x4 kernel
        - Normalize -> LeakyReLU
        - Conv2d: 8 * ndf -> 16 * ndf, 4x4 kernel
        - Normalize -> LeakyReLU
        ...

    Final:
        - Conv2D: 16 * ndf -> 1, 4x4 kernel
        - Sigmoid
    '''

    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 gpu_ids=[],
                 conv=nn.Conv2d):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d) or (
                norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (
                norm_layer == nn.InstanceNorm3d)

        kw = 4
        padw = 1
        sequence = [
            conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            # Conv2d: ndf -> 8 * ndf, 4x4 kernel size on first, the next 2
            # are 8 * ndf -> 8 * ndf, 4x4 kernel, then all subsequent ones
            # are 2 ** n * ndf -> 2 ** (n+1) * ndf, 4x4 kernel
            # each Conv followed with a norm_layer and LeakyReLU
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                conv(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final Conv2d: 2 ** n * ndf -> 1, 4x4 kernels
        sequence += [
            conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data,
                                            torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    ''' Pixel-based rather than patches.

    Architecture:
        - Conv2d: inputC -> ndf, 1x1 kernels
        - Normalize -> Leaky ReLU
        - Conv2d: ndf -> ndf * 2, 1x1 kernels
        - Normalize -> Leaky ReLU
        - Conv2d: ndf * 2 -> 1, 1x1 kernels
        - Sigmoid
    '''

    def __init__(self,
                 input_nc,
                 ndf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 gpu_ids=[],
                 conv=nn.Conv2d):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d) or (
                norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d) or (
                norm_layer == nn.InstanceNorm3d)

        self.net = [
            conv(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            conv(
                ndf,
                ndf * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            conv(ndf * 2, 1, kernel_size=1, strid=1, padding=0, bias=use_bias)
        ]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data,
                                            torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class UnetCNNGenerator(nn.Module):
    ''' Defines the UNet-CNN from that one paper.
    Architecture contains downsampling/upsampling operations, with Unet
    connectors.

    - Outermost Unet:
        - Conv2d: InputC -> 64, 4x4 kernels

    - Outer Unets:
        - Leaky ReLU (iv)
        - Conv2d: 64 -> 128, 4x4 kernels
        - Leaky ReLU (iii)
        - Conv2d: 128 -> 256, 4x4 kernels
        - Leaky ReLU (ii)
        - Conv2d: 256 -> 512, 4x4 kernels

    - Intermediate Unets (x num_downs):
        - Leaky ReLU (i)
        - Conv2d: 512 -> 512, 4x4 kernels

    - Inner Unet:
        - Leaky ReLU (a)
        - Conv2d: 512 -> 512, 4x4 kernels
        - ReLU
        - Deconv2d: 512 -> 512, 4x4 kernels
        - Normalize --> Connect to (a)

    - Intermediate Unets continued:
        - ReLU
        - Deconv2d: 1024 -> 512, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (i)

    - Outer Unets:
        - ReLU
        - Deconv2d: 512 -> 256, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (ii)
        - ReLU
        - Deconv2d: 256 -> 128, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iii)
        - ReLU
        - Deconv2d: 128 -> 64, 4x4 kernels
        - Normalize (-> Dropout ->) --> Connect to (iv)

    - Outermost Unet:
        - ReLU
        - Deconv2d: 128 -> outC, 4x4 kernels
        - Tanh
    '''

    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 gpu_ids=[],
                 conv=nn.Conv2d,
                 deconv=nn.ConvTranspose2d):
        '''
        Args:
            num_downs:  number of downsamplings in the Unet.
        '''
        super(UnetCNNGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # blocks are built recursively starting from innermost moving out
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            conv=conv,
            deconv=deconv)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                conv=conv,
                deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)
        unet_block = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            conv=conv,
            deconv=deconv)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
