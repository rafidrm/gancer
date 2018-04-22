def create_model(opt):
    model = None
    print(opt.model)
    # if opt.model == 'cycle_gan':
    #     assert (opt.dataset_mode == 'unaligned')
    #     from .cycle_gan_model import CycleGANModel
    #     raise NotImplementedError('I have not implemented CycleGAN yet.')
    #     model = CycleGANModel()
    if opt.model == 'pix2pix':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'slice')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'vox2vox':
        assert (opt.dataset_mode == 'voxel')
        from .vox2vox_model import Vox2VoxModel
        model = Vox2VoxModel()
    elif opt.model == 'unetcnn':
        assert (opt.dataset_mode == 'slice')
        from .unetcnn_model import UnetCNNModel
        model = UnetCNNModel()
    elif opt.model == 'test':
        assert (opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [{}] not recognized".format(opt.model))
    model.initialize(opt)
    print("model [{}] was created".format(model.name()))
    return model
