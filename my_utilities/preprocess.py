import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def preprocess(
    *,
    main_data_path:str,
    processed_data_path:str,
    mask_to_brain_wanted_ratio:float,
    modalities:tuple,
    crop_size:list[int]
):
    for patient in tqdm(sorted(os.listdir(main_data_path))):
        if patient.split('.')[-1] == 'csv':
            continue
        path = os.path.join(main_data_path, patient, patient)
        path += '_'
        # [channel(number of modalities), height, width, depth] 
        image = np.stack([np.array(nib_load(path + modality + '.nii.gz'), dtype='float32') for modality in modalities], axis = 0) 
        # height, width, depth
        mask = np.array(nib_load(path + 'seg.nii.gz'), dtype = 'uint8')
        
        image, mask = crop_zero_margin(image, mask, crop_size)
        # image.shape (4, 192, 192, 144)
        # mask.shape (192, 192, 144)
        for idx in range(image.shape[-1]):
            img = image[...,idx]
            msk = mask[...,idx]

            if (msk==0).sum() == 0 or (msk==1).sum() == 0 or (msk==2).sum() == 0 or (msk==4).sum() == 0:
                continue
            
            img_vol = (img[0,...]>0).sum()
            msk_vol = (msk>0).sum()
            mask_to_brain_ratio = msk_vol / (img_vol + 1e-8)
            if mask_to_brain_ratio < mask_to_brain_wanted_ratio:
                continue

            img = img.astype(np.float32)
            msk = msk.astype(np.uint8)
            img = min_max_normalize(img)
            msk = reEncodeMaskZeroBase(msk)
            msk = one_hot_encode(msk, num_class=4)
            # img.shape (4, 192, 192)
            # msk.shape (4, 192, 192)
            
            # img_p = img * 255
            # msk_p = msk.argmax(axis=0)
            # brain_vol = (img_p>0).sum()
            # mask_vol = (msk_p>0).sum()
            # smooth = 1e-8
            # mask_to_brain_ratio = (mask_vol)/(brain_vol+smooth)
            # mask_to_brain_ratio *= 100


            out_path = os.path.join(processed_data_path, patient)
            out_file = patient + "_s_" + str(idx) + "_r_{0}".format(mask_to_brain_ratio)
            out_file = out_file[:-1]

            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            np.save(os.path.join(out_path,out_file+'_img'), img)
            np.save(os.path.join(out_path,out_file+"_msk"), msk)


def nib_load(file_name):
    """ reads nii.gz file and returns numpy array

    Parameters
    ----------
    file_name: str
        full path of nii.gz file
    """
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def reEncodeMaskZeroBase(mask):
    """ Encodes labels zero base

    Parameters
    ----------
    mask: numpy array
        2D mask to be encoded zero base
    """
    labelencoder = LabelEncoder()

    h, w = mask.shape

    maskـreshaped = mask.reshape(-1)
    mask_reshaped_encoded = labelencoder.fit_transform(maskـreshaped)
    mask__encoded = mask_reshaped_encoded.reshape(h, w)

    return mask__encoded


def one_hot_encode(mask, num_class=4):
    """ One hot encode the mask

    Parameters
    ----------
    mask: numpy array
        2D mask to be one hot encoded
    num_class: int, optional
        number of one hot encode classes
    """
    assert mask.ndim == 2
    shape = list(mask.shape)
    shape.insert(0, num_class)
    # if shape was (192, 192) it becames (4, 192, 192)
    categorical = np.zeros(tuple(shape))
    categorical[0, ...] = (mask == 0)
    categorical[1, ...] = (mask == 1)
    categorical[2, ...] = (mask == 2)
    categorical[3, ...] = (mask == 3)

    return categorical

def min_max_normalize(input):
    #channel, height, width
    for idx in range(input.shape[0]):
        if (np.max(input[idx, ...]) - np.min(input[idx, ...])) > 0:
            input[idx, ...] = (input[idx, ...] - np.min(input[idx, ...])) / \
                (np.max(input[idx, ...]) - np.min(input[idx, ...]))
    return input
    
        
##########################################################
################### Crop zero margin #####################


def crop_zero_margin(image, mask, size=[192, 192, 144]):
    box_min, box_max = get_box(image[0, ...], 0)
    index_min, index_max = make_box(
        image[0, ...], box_min, box_max, size)
    tmp = size.copy()
    tmp.insert(0, image.shape[0])
    tmp2 = size.copy()
    # tmp2.insert(0, mask.shape[0])
    img = np.zeros(tuple(tmp))
    msk = np.zeros(tuple(tmp2))
    for i in range(image.shape[0]):
        img[i, ...] = crop_with_box(
            image[i, ...], index_min, index_max)
    msk = crop_with_box(mask, index_min, index_max)
    return img, msk


def get_box(image, margin):
    """
        Cut out, deduct the area with no pixels around the image. 
        The margin is the reserved area parameter, such as: margin=3, 
        which means that there are three pixels around the pixel. 
    """
    shape = image.shape
    # What is returned is 3 arrays, corresponding to the subscripts of the three dimensions.
    nonindex = np.nonzero(image)

    index_min = []
    index_max = []

    for i in range(len(shape)):
        index_min.append(nonindex[i].min())
        index_max.append(nonindex[i].max())

    if margin > 0:
        margin = [margin] * len(shape)
        # Expand margin areas
        for i in range(len(shape)):
            index_min[i] = max(index_min[i] - margin[i], 0)
            index_max[i] = min(index_max[i] + margin[i], shape[i]-1)

    # print(index_min)
    # print(index_max)
    return index_min, index_max


def make_box(image, index_min, index_max, data_box):
    """
        Cut out the image, use the subscript obtained by get_box(). 
    """
    shape = image.shape

    for i in range(len(shape)):

        # print('before index[%s]: '%i, index_min[i], index_max[i])

        # Expand or reduce the box according to data_box.
        mid = (index_min[i] + index_max[i])/2
        index_min[i] = mid - data_box[i]/2
        index_max[i] = mid + data_box[i]/2

        flag = index_max[i] - shape[i]
        if flag > 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag

        flag = index_min[i]
        if flag < 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag

        # print('index[%s]: '%i, index_min[i], index_max[i])

        if index_max[i] - index_min[i] != data_box[i]:
            index_max[i] = index_min[i] + data_box[i]

        index_max[i] = int(index_max[i])
        index_min[i] = int(index_min[i])

        # print('after index[%s]: '%i, index_min[i], index_max[i])
    return index_min, index_max


def crop_with_box(image, index_min, index_max):
    """
        Divide the image according to the box. 
    """
    # return image[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]
    x = index_max[0] - index_min[0] - image.shape[0]
    y = index_max[1] - index_min[1] - image.shape[1]
    z = index_max[2] - index_min[2] - image.shape[2]
    img = image
    img1 = image
    img2 = image

    if x > 0:
        img = np.zeros((image.shape[0]+x, image.shape[1], image.shape[2]))
        img[x//2:image.shape[0]+x//2, :, :] = image[:, :, :]
        img1 = img

    if y > 0:
        img = np.zeros((img1.shape[0], img1.shape[1]+y, img1.shape[2]))
        img[:, y//2:image.shape[1]+y//2, :] = img1[:, :, :]
        img2 = img

    if z > 0:
        img = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2]+z))
        img[:, :, z//2:image.shape[2]+z//2] = img2[:, :, :]

    return img[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]

##########################################################






if __name__ == '__main__':
    preprocess(
        main_data_path='/media/idham/SpaceX/333_workspace/deeplearning/unet/MICCAI_BraTS2020_TrainingData',
        crop_size=[192, 192, 144],
        mask_to_brain_wanted_ratio=0.2,
        modalities=('flair', 't1ce', 't1', 't2'),
        processed_data_path='/media/idham/SpaceX/333_workspace/my_lab/my_processed_data'
    )