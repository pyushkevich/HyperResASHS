import SimpleITK as sitk
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import save_json, join
from scipy.ndimage import label as ndi_label
from picsl_c3d import Convert3D


def flip_image(input_path, output_path, flip_axis):
    image_itk = sitk.ReadImage(input_path)
    image_array = sitk.GetArrayFromImage(image_itk)

    # flip it
    image_array = np.flip(image_array, axis=flip_axis)

    # save it
    new_itk = sitk.GetImageFromArray(image_array)
    new_itk.SetOrigin(image_itk.GetOrigin())
    new_itk.SetDirection(image_itk.GetDirection())
    new_itk.SetSpacing(image_itk.GetSpacing())

    sitk.WriteImage(new_itk, output_path)


def convert_each_ground_truth_file_as_continuous(convert_file, nnunet_raw_path):
    # set convert file
    convert_content = [ele_.strip('\n') for ele_ in open(convert_file)]

    # read the ground truth
    gt_path = os.path.join(nnunet_raw_path, 'labelsTr')
    file_list = os.listdir(gt_path)

    # set output path
    for file_ in file_list:
        refseg_path = os.path.join(gt_path, file_)
        replaced_path = os.path.join(gt_path, file_)

        # read the source image
        image_itk = sitk.ReadImage(refseg_path)
        image_array = sitk.GetArrayFromImage(image_itk)

        # create a totally new array
        new_array = np.zeros(image_array.shape)

        # replace each subregion as target label
        for tlt_ in convert_content:
            old_id = int(tlt_.split(',')[0])
            new_id = int(tlt_.split(',')[1])
            new_array = new_array + ((image_array == old_id) + 0) * int(new_id)
            print('replace {} as {}'.format(old_id, new_id))

        # generate new nii file
        new_itk = sitk.GetImageFromArray(new_array)
        new_itk.SetOrigin(image_itk.GetOrigin())
        new_itk.SetDirection(image_itk.GetDirection())
        new_itk.SetSpacing(image_itk.GetSpacing())
        sitk.WriteImage(new_itk, replaced_path)

        print('finish and cover the {}'.format(replaced_path))


def save_label_mapping_to_txt(snap_label_path, save_path):
    snap_content = [ele_.strip('\n') for ele_ in open(snap_label_path) if ele_[0] != '#']

    # new and old label
    convert_list = []
    for ii, ele_ in enumerate(snap_content):
        # get curr id
        curr_id = None
        curr_ele_all_list = ele_.split(' ')
        for kkt_ in curr_ele_all_list:
            if kkt_ != '':
                curr_id = kkt_
                break
        
        # get curr label name
        curr_label = ele_.strip('"').split('"')[-1]
        # convert_list.append('{},{},{}'.format(curr_id, ii, curr_label))  # use continuous labels
        convert_list.append('{},{},{}'.format(curr_id, curr_id, curr_label))  # use original labels
    
    with open(save_path, 'w') as f_:
        for ele_ in convert_list:
            f_.write(ele_ + '\n')


def make_nnunet_dataset_json(convert_file, output_folder):
    # read convert file
    convert_content = [ele_.strip('\n') for ele_ in open(convert_file)]

    # count the number of training set
    labeltr_folder = os.path.join(output_folder, 'labelsTr')
    numtraining = len(os.listdir(labeltr_folder))

    # make different dict
    ## channel name
    modality_list = []
    file_list = os.listdir(os.path.join(output_folder, 'imagesTr'))
    for file_ in file_list:
        modality_ = file_.split('_')[2].strip('.nii.gz')
        if modality_ not in modality_list:
            modality_list.append(modality_)
    channel_names = {}
    for ii in range(len(modality_list)):
        channel_names[str(ii)] = modality_list[ii]

    ## specific label
    labels = {}
    for ele_ in convert_content:
        new_id = ele_.split(',')[1]
        label_name = ele_.split(',')[2]

        if label_name == 'Clear Label':
            label_name = 'background'
        
        if label_name in labels:
            if labels[label_name] != int(new_id):
                raise ValueError('Label conflicts!')
        else:
            labels[label_name] = int(new_id)
    
    ## other information
    dataset_json = {
        'channel_names': channel_names,  
        'labels': labels,
        'numTraining': numtraining,
        'file_ending': ".nii.gz",
    }

    # fill those skipped labels
    fill_labels(dataset_json)
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


def fill_labels(d):
    labels = d["labels"]

    value_to_key = {v: k for k, v in labels.items()}
    max_val = max(labels.values())
    
    new_labels = {}
    for i in range(max_val + 1):
        if i in value_to_key:
            new_labels[value_to_key[i]] = i
        else:
            new_labels[f"empty_label_{i}"] = i
    
    d["labels"] = new_labels
    return d

def linear_resample_to_spacing_using_itkimage(image_primary, new_spacing):
    original_spacing = image_primary.GetSpacing()
    original_size = image_primary.GetSize()

    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    new_spacing = [ospc * osz / nsz for ospc, osz, nsz in zip(original_spacing, original_size, new_size)]

    c3d = Convert3D()
    c3d.push(image_primary)
    c3d.execute(f"-resample {new_size[0]}x{new_size[1]}x{new_size[2]}")
    resampled_image = c3d.peek(-1)

    return resampled_image