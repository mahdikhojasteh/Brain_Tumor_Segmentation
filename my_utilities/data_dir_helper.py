import os
import pathlib
from pathlib import Path
import numpy as np


def generate_processed_data_dirs(*,
                                 preprocessed_path: str = '/media/idham/SpaceX/333_workspace/deeplearning/unet/A_BrainSeg/data',
                                 check_sync: bool = False,
                                 output_name: str = 'stacked_img_msk_data_dirs'
                                 ):
    processed_data_root = Path(preprocessed_path)

    find = '*' + os.sep + '*img.npy'
    img_dirs = list(processed_data_root.glob(find))
    find = '*' + os.sep + '*msk.npy'
    msk_dirs = list(processed_data_root.glob(find))

    img_dirs.sort(key=lambda x: (int(str(x.resolve()).split(os.sep)[-2].split('_')[-1]),
                                 int(str(x.resolve()).split(os.sep)[-1].split('_')[4])))
    msk_dirs.sort(key=lambda x: (int(str(x.resolve()).split(os.sep)[-2].split('_')[-1]),
                                 int(str(x.resolve()).split(os.sep)[-1].split('_')[4])))

    if check_sync:
        """ checking img_dirs and msk_dirs are completely sync
        """
        assert len(img_dirs) == len(msk_dirs)

        for i in range(len(img_dirs) * 100):
            idx = np.random.randint(0, len(img_dirs), 1)[0]
            img_dir = img_dirs[idx]
            msk_dir = msk_dirs[idx]

            img_patient1 = str(img_dir).split(os.sep)[-2].split('_')[-1]
            img_patient2 = str(img_dir).split(os.sep)[-1].split('_')[2]
            img_slice = str(img_dir).split(os.sep)[-1].split('_')[4]
            img_ratio = str(img_dir).split(os.sep)[-1].split('_')[6]

            msk_patient1 = str(msk_dir).split(os.sep)[-2].split('_')[-1]
            msk_patient2 = str(msk_dir).split(os.sep)[-1].split('_')[2]
            msk_slice = str(msk_dir).split(os.sep)[-1].split('_')[4]
            msk_ratio = str(msk_dir).split(os.sep)[-1].split('_')[6]

            assert img_patient1 == img_patient2 == msk_patient1 == msk_patient2
            assert img_slice == msk_slice

    stacked_img_msk_data_dirs = np.stack([img_dirs, msk_dirs], axis=1)

    save_path = Path(Path('.'), output_name)

    np.save(save_path, stacked_img_msk_data_dirs)


def get_processed_data_path(*,
                            stacked_data_path: str = Path(
                                Path('.'), 'stacked_img_msk_data_dirs.npy'),
                            is_windows=False
                            ):
    if is_windows:
        pathlib.PosixPath = pathlib.WindowsPath
    return np.load(stacked_data_path, allow_pickle=True)
