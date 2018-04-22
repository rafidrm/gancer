import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    ''' Gets called by CustomDatasetDataLoader.initialize(). dataset_mode is
    by default unaligned. Dataset has generic structure, inputs are coming
    from opts. Aligned, Unaligned are for A->B (i.e., image-to-image transfer
    type problems, whereas Single is for z->A problems (and testing).
    '''
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif opt.dataset_mode == 'slice':
        from data.slice_dataset import SliceDataset
        dataset = SliceDataset()
    elif opt.dataset_mode == 'voxel':
        from data.voxel_dataset import VoxelDataset
        dataset = VoxelDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):

    ''' Inherited from Base. Carries functions initialize and load_data '''

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        # Torch Dataloader combines a dataset and sampler, provides settings.
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  # dataset class
            batch_size=opt.batchSize,  # how many samples/batch to load
            shuffle=not opt.serial_batches,  # reshuffle per epoch
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:  # if more data than want to use
                break
            yield data
