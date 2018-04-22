import os
import pudb
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.io import loadmat, savemat


def patientwise_splitting(train, test, img_list):
    patient_ids = [f.split('_')[1] for f in img_list]
    patient_ids = list(set(patient_ids))
    train_ids, test_ids = train_test_split(patient_ids, train_size=train)
    print('Train_ids=')
    print(train_ids)
    test_ids, val_ids = train_test_split(patient_ids, train_size=test)
    print('Test_ids=')
    print(test_ids)
    print('Val_ids=')
    print(val_ids)
    x_train = []
    x_test = []
    x_val = []
    for fname in img_list:
        patient_id = fname.split('_')[1]
        if patient_id in train_ids:
            x_train.append(fname)
        elif patient_id in test_ids:
            x_test.append(fname)
        elif patient_id in val_ids:
            x_val.append(fname)
        else:
            raise ValueError(
                'file [{}] is not in train-test-split'.format(fname))
    return x_train, x_test, x_val


def instructionwise_splitting(split_instructions, img_list):
    instr_mat = loadmat(split_instructions)
    train_ids = instr_mat['train_pats']
    train_ids = list(train_ids[0])
    test_ids = instr_mat['test_pats']
    test_ids = list(test_ids[0])
    val_ids = instr_mat['valid_pats']
    val_ids = list(val_ids[0])
    x_train = []
    x_test = []
    x_val = []
    for fname in img_list:
        patient_id = int(fname.split('_')[1].split('.')[0])
        if patient_id in train_ids:
            x_train.append(fname)
        elif patient_id in test_ids:
            x_test.append(fname)
        elif patient_id in val_ids:
            x_val.append(fname)
        else:
            raise ValueError(
                'file [{}] is not in train-test-split'.format(fname))
    return x_train, x_test, x_val


def ttsplit_and_copy(aaron_dir,
                     data_dir,
                     train,
                     test,
                     split_by_patient=False,
                     split_instructions=None):
    img_list = os.listdir(aaron_dir)
    if split_by_patient:
        x_train, x_test, x_val = patientwise_splitting(train, test, img_list)
    elif split_instructions:
        x_train, x_test, x_val = instructionwise_splitting(
            split_instructions, img_list)
    else:
        x_train, x_test = train_test_split(img_list, train_size=train)
        x_test, x_val = train_test_split(x_test, train_size=test)
        print('Train_ids=')
        print(x_train)
        print('Test_ids=')
        print(x_test)
        print('Val_ids=')
        print(x_val)
    for fname in tqdm(x_train):
        shutil.copy(
            os.path.join(aaron_dir, fname),
            os.path.join(data_dir, 'train', fname))
    for fname in tqdm(x_test):
        shutil.copy(
            os.path.join(aaron_dir, fname),
            os.path.join(data_dir, 'test', fname))
    for fname in tqdm(x_val):
        shutil.copy(
            os.path.join(aaron_dir, fname), os.path.join(
                data_dir, 'val', fname))


def move_to_cancerGAN(aaron_dir, data_dir, new_dir=None, train=0.6, test=0.5):
    ''' Taking aaron's jpegs and parsed them. '''
    img_list = os.listdir(aaron_dir)
    num_files = len(img_list)
    if new_dir is not None:
        for i in tqdm(range(num_files)):
            shutil.copy(
                os.path.join(aaron_dir, img_list[i]),
                os.path.join(new_dir, '{}.jpg'.format(i + 1)))
            aaron_dir = new_dir

    ttsplit_and_copy(aaron_dir, data_dir, train, test)


def collect_parse_mat_slices(aaron_dir,
                             data_dir,
                             new_dir=None,
                             train=0.6,
                             test=0.5,
                             split_by_patient=False,
                             with_copy=False):
    ''' Takes 2 folders, merges them appropriately, then ttsplit and resave.'''
    if new_dir is not None and len(aaron_dir) == 2:
        clin_dir == aaron_dir[0]
        ct_dir = aaron_dir[1]
        clin_list = os.listdir(clin_dir)
        ct_list = os.listdir(ct_dir)
        for clinFile in tqdm(clin_list):
            if os.path.isfile(os.path.join(ct_dir, clinFile)):
                try:
                    clin = loadmat(os.path.join(clin_dir, clinFile))
                    ct = loadmat(os.path.join(ct_dir, clinFile))
                    mDict = {'dMs': clin['dMs'], 'iMs': ct['iMs']}
                    saveFile = os.path.join(new_dir, clinFile)
                    savemat(saveFile, mDict)
                except:
                    pass
            else:
                print('File [{}] does not exist in CT directory'.format(
                    clinFile))
                aaron_dir = new_dir

    ttsplit_and_copy(aaron_dir, data_dir, train, test, split_by_patient)


def test_corruptions(aaron_dir, new_dir):
    img_list = os.listdir(aaron_dir)
    for img_file in tqdm(img_list):
        try:
            img = loadmat(os.path.join(aaron_dir, img_file))
            shutil.copy(
                os.path.join(aaron_dir, img_file),
                os.path.join(new_dir, img_file))
            # imgDict = {'dMs': img['dMs'], 'iMs': img['iMs']}
            # savefile = os.path.join(new_dir, img_file)
            # savemat(savefile, imgDict)
        except:
            print('Failed on file [{}]'.format(img_file))


if __name__ == '__main__':
    twoDee = False

    if twoDee:
        aaron_dir = os.path.join('Aaron', 'MedPhys_Gan_4mm_2D_noCT')
        new_dir = 'merged_2d_noct'
        data_dir = '/home/rm/Python/cancerGAN/cancerGAN/datasets/cancer_noct'
        split_by_patient = True

        test_corruptions(aaron_dir, new_dir)
        collect_parse_mat_slices(new_dir, data_dir, split_by_patient=True)
    else:
        # 3-D
        aaron_dir = os.path.join('Aaron', 'MedPhys_Gan_4mm_3D')
        data_dir = '/home/rm/Python/cancerGAN/cancerGAN/datasets/voxels_128'
        new_dir = 'merged_3d'
        split_instructions = os.path.join('Aaron', 'pat_cats.mat')
        # test_corruptions(aaron_dir, new_dir=new_dir)
        ttsplit_and_copy(
            new_dir,
            data_dir,
            train=0.6,
            test=0.4,
            split_by_patient=False,
            split_instructions=split_instructions)
