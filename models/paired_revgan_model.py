import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F


class PairedRevGANModel(BaseModel):
    ''' Paired 2D-RevGAN model '''
    def name(self):
        return 'PairedRevGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='instance')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_AB', 'G_AB', 'D_BA', 'G_BA']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A_enc', 'G_B_enc', 'G_core', 'G_A_dec', 'G_B_dec', 'D_AB', 'D_BA']
        else:  # during test time, only load Gs
            self.model_names = ['G_A_enc', 'G_B_enc', 'G_core', 'G_A_dec', 'G_B_dec']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A_enc = networks.define_G_enc(opt.input_nc, opt.output_nc,
                                                opt.ngf, opt.which_model_netG, opt.norm,
                                                opt.init_type, opt.init_gain, self.gpu_ids, n_downsampling=opt.n_downsampling)

        self.netG_core = networks.define_G_core(opt.input_nc, opt.output_nc,
                                                opt.ngf, opt.which_model_netG, opt.norm, opt.use_naive,
                                                opt.init_type, opt.init_gain, self.gpu_ids,
                                                invertible=True, n_downsampling=opt.n_downsampling,
                                                coupling=opt.coupling)

        self.netG_A_dec = networks.define_G_dec(opt.input_nc, opt.output_nc,
                                                opt.ngf, opt.which_model_netG, opt.norm,
                                                opt.init_type, opt.init_gain, self.gpu_ids, not opt.no_output_tanh, n_downsampling=opt.n_downsampling)

        self.netG_B_enc = networks.define_G_enc(opt.output_nc, opt.input_nc,
                                                opt.ngf, opt.which_model_netG, opt.norm, 
                                                opt.init_type, opt.init_gain, self.gpu_ids, n_downsampling=opt.n_downsampling)

        self.netG_B_dec = networks.define_G_dec(opt.output_nc, opt.input_nc,
                                                opt.ngf, opt.which_model_netG, opt.norm, 
                                                opt.init_type, opt.init_gain, self.gpu_ids, not opt.no_output_tanh, n_downsampling=opt.n_downsampling)

        self.netG_A = lambda x: self.netG_A_dec(self.netG_core(self.netG_A_enc(x)))
        self.netG_B = lambda x: self.netG_B_dec(self.netG_core(self.netG_B_enc(x), inverse=True))
        

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_AB = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_BA = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_BA_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A_enc.parameters(),
                                                                self.netG_A_dec.parameters(),
                                                                self.netG_core.parameters(),
                                                                self.netG_B_enc.parameters(),
                                                                self.netG_B_dec.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_AB.parameters(),
                                                                self.netD_BA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.fake_A = self.netG_B(self.real_B)

    def backward_D(self):
        # Fake AB
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        loss_D_fake_AB = self.criterionGAN(self.netD_AB(fake_AB.detach()), False)
        # Real AB
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        loss_D_real_AB = self.criterionGAN(self.netD_AB(real_AB), True)

        # Fake BA
        fake_BA = self.fake_BA_pool.query(torch.cat((self.real_B, self.fake_A), 1))
        loss_D_fake_BA = self.criterionGAN(self.netD_BA(fake_BA.detach()), False)
        # Real BA
        real_BA = torch.cat((self.real_B, self.real_A), 1)
        loss_D_real_BA = self.criterionGAN(self.netD_BA(real_BA), True)

        # AB
        self.loss_D_AB = loss_D_fake_AB + loss_D_real_AB

        # BA
        self.loss_D_BA = loss_D_fake_BA + loss_D_real_BA

        # Combined loss
        loss_D = (self.loss_D_AB + self.loss_D_BA) * 0.5

        # backward
        if self.opt.grad_reg > 0.0:
            loss_D.backward(retain_graph=True)
            Lgrad = torch.cat([x.grad.view(-1) for x in [*self.netD_AB.parameters(), *self.netD_BA.parameters()]])
            loss_D = loss_D - self.opt.grad_reg * (0.5 * torch.norm(Lgrad) / len(Lgrad))
            loss_D.backward()
        else:
            loss_D.backward(retain_graph=True)
        return loss_D

    def backward_G(self):
        # G_A
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G_AB = self.criterionGAN(self.netD_AB(fake_AB), True)
        if self.opt.grad_reg > 0.0:
            self.loss_G_AB.backward(retain_graph=True)
            Lgrad = torch.cat([x.grad.view(-1) for x in [*self.netG_A_enc.parameters(),
                                                         *self.netG_core.parameters(),
                                                         *self.netG_A_dec.parameters()]])
            self.loss_G_AB = self.loss_G_AB - self.opt.grad_reg * (0.5 * torch.norm(Lgrad) / len(Lgrad))

        # G_B
        fake_BA = torch.cat((self.real_B, self.fake_A), 1)
        self.loss_G_BA = self.criterionGAN(self.netD_BA(fake_BA), True)
        if self.opt.grad_reg > 0.0:
            self.loss_G_BA.backward(retain_graph=True)
            Lgrad = torch.cat([x.grad.view(-1) for x in [*self.netG_B_enc.parameters(),
                                                         *self.netG_core.parameters(),
                                                         *self.netG_B_dec.parameters()]])
            self.loss_G_BA = self.loss_G_BA - self.opt.grad_reg * (0.5 * torch.norm(Lgrad) / len(Lgrad))

        # L1
        self.loss_G_L1_A = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1
        self.loss_G_L1_B = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Combined loss
        self.loss_G = self.loss_G_AB + self.loss_G_BA + self.loss_G_L1_A + self.loss_G_L1_B

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_AB, self.netD_BA], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_AB, self.netD_BA], True)
        for _ in range(self.opt.D_rollout):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
