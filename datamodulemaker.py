'''
======================================================================================================================
BeeID - The bee reidentification data set MODULE MAKER
======================================================================================================================
  .--.               .--.
 /    `.   o   o   .'    \
 \      \   \ /   /      /
 /\_     \ .-"-. /     _/\		WHAT IS THE SOCIETY WE WISH TO PROTECT?
(         V ^ ^ V         )		IS IT THE SOCIETY OF COMPLETE SURVEILLANCE FOR THE COMMONWEALTH?
 \_      _| 9_9 |_      _/		IS THIS THE WEALTH WE SEEK TO HAVE IN COMMON
  `.    //\__o__/\\    .'		OPTIMAL SECURITY
    `._//\=======/\\_.'			AT THE COST OF MAXIMAL SURVEILLANCE?
     /_/ /\=====/\ \_\
       _// \===/ \\_
      /_/_//`='\\_\_\ hjw		- TOM STOPPARD
        /_/     \_\
======================================================================================================================
contact: mark.schutera@kit.edu / mark.schutera@mailbox.org
======================================================================================================================
'''


import os
import json
import cv2
import tarfile
import shutil
import urllib.request
import urllib.error
import numpy as np

filenames = ['S1', 'S2', 'S3', 'S4', 'S5']


def get_data(filename, urlname):
    try:
        print("Downloading data ", filename, '..')
        urllib.request.urlretrieve("https://beepositions.unit.oist.jp/" + urlname, filename)
    except urllib.error.URLError:
        print("The data is no longer available through this link ..")
        print("Please contact k.bozek@uni-koeln.de for guidance and to point to the new data source.")
        exit()


def extract_all_files(sequence_names):
    # Create folder structure
    try:
        os.mkdir('beeid_data')
        os.mkdir('beeid_data/raw')
        os.mkdir('beeid_data/images')
    except WindowsError:
        print('WARNING: You already set up your folder structure')

    for filename in sequence_names:
        # urlname = filename + '.tgz'
        filename += '.gz'
        #get_data(filename, urlname)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

        os.remove(filename)


# create frames from mp4
def mp4_to_frames(mp4_name):
    path = './' + mp4_name + '/' + mp4_name + '.mp4'
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite('beeid_data/raw/' + mp4_name + '_{:04d}.jpg'.format(count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read frame: ', count)
        count += 1


def progress_checker():
    path = 'beeid_data/images/'
    images_dir = sorted(os.listdir(path))

    if len(images_dir) == 0:
        print('Generating BeeID dataset ..')
        return 0, 0, 0
    else:
        latest_file = images_dir[-1]
        bee_id, sequence, _ = str.split(latest_file, '_')

        # adjust trajectory_id based on sequence
        trajectory_id = 0
        if sequence == '01':
            trajectory_id = int(bee_id)
        if sequence == '02':
            trajectory_id = int(bee_id)-len(os.listdir('./S1/trajectories/'))-1
        if sequence == '03':
            trajectory_id = int(bee_id)-len(os.listdir('./S1/trajectories/'))-1\
                            - len(os.listdir('./S2/trajectories/'))
        if sequence == '04':
            trajectory_id = int(bee_id)-len(os.listdir('./S1/trajectories/'))-1\
                            - len(os.listdir('./S2/trajectories/'))\
                            - len(os.listdir('./S3/trajectories/'))
        if sequence == '05':
            trajectory_id = int(bee_id)-len(os.listdir('./S1/trajectories/'))-1\
                            - len(os.listdir('./S2/trajectories/'))\
                            - len(os.listdir('./S3/trajectories/'))\
                            - len(os.listdir('./S4/trajectories/'))

        print('Continue to generate BeeID dataset:')
        print('Sequence: ', str(int(sequence)), ', bee_id: ', bee_id, ', trajectory_id: ', trajectory_id)
        return int(bee_id), int(sequence)-1, trajectory_id


# crop images
def crop_image(sequence, frame, centerpoint, bee_id, boxedge=45, frame_skip=10):
    # crop detections from raw frames into '{:04d}_{:01d}_{:04d}.jpg'.format(bee_id, sequence_id, image_id)
    # load image
    if frame % frame_skip == 0:
        path = 'beeid_data/raw/S' + str(sequence) + '_{:04d}'.format(frame) + '.jpg'
        # print(path)
        img = cv2.imread(path)
        # print(centerpoint[1]-boxedge, centerpoint[1]+boxedge, centerpoint[0]-boxedge, centerpoint[0]+boxedge)
        cropped_img = img[max(0, centerpoint[1]-boxedge):centerpoint[1]+boxedge, centerpoint[0]-boxedge:centerpoint[0]+boxedge]
        cv2.imwrite('beeid_data/images/' + '{:08d}_{:02d}_{:04d}.jpg'.format(int(bee_id), int(sequence), int(frame)),
                    cropped_img)


def get_trajectory_array(path):
    file_names = os.listdir(path)
    trajectory_array = np.zeros(len(file_names), dtype=object)
    for i, filename in enumerate(file_names):
        with open(os.path.join(path, filename)) as f:
            trajectory_array[i] = np.array([[int(j) for j in j.strip().split(',')] for j in f.readlines()])
    return trajectory_array


def crop_image2(img, sequence, frame, centerpoint, bee_id, boxedge=45):
    cropped_img = img[max(0, centerpoint[1]-boxedge):centerpoint[1]+boxedge, centerpoint[0]-boxedge:centerpoint[0]+boxedge]
    cv2.imwrite('beeid_data/images/' + '{:08d}_{:02d}_{:04d}.jpg'.format(int(bee_id), int(sequence), frame), cropped_img)


def generate_crops2(sequence_names, frame_skip=10):
    frames_all = os.listdir('beeid_data/raw/')
    bee_id_global = 0
    for filename in sequence_names:
        trajectory_path = './' + filename + '/trajectories/'
        trajectory_array = get_trajectory_array(trajectory_path)
        frames_ = [i for i in frames_all if filename in i][::frame_skip]
        for frame in frames_:
            print('beeid_data/raw/'+frame)
            image = cv2.imread(os.path.join('./beeid_data/raw/' + frame))
            frame_no = 0 if frame[3:-4] == '0000' else frame[3:-4].lstrip('0')
            # bee_id = 0
            for traj_file in range(trajectory_array.shape[0]):
                idx = np.where(trajectory_array[traj_file][:,0] == int(frame_no))
                if not idx[0].shape[0] == 0:
                    traj_data = trajectory_array[traj_file][idx].squeeze()
                    center_point = [traj_data[1], traj_data[2]]
                    crop_image2(image, filename[-1], traj_data[0], center_point, bee_id_global + traj_file)
                # bee_id += 1
        bee_id_global += len(os.listdir(trajectory_path))
                           

# create meta.json
def generate_crops(sequence_names):
    # check progress
    bee_id, sequence, trajectory_id = progress_checker()

    # load tracklet information
    for filename in sequence_names[sequence:]:
        trajectory_path = './' + filename + '/trajectories/'
        for trajectory in sorted(os.listdir(trajectory_path))[trajectory_id:]:
            print(trajectory_path + trajectory)
            with open(trajectory_path + trajectory) as trs:
                tr = trs.readline().strip().split(',')
                while tr:
                    try:
                        centerpoint = [int(tr[1]), int(tr[2])]
                        crop_image(filename[-1], int(tr[0]), centerpoint, bee_id)
                        tr = trs.readline().strip().split(',')
                    except IndexError:
                        break
            bee_id += 1


def generate_meta_json():
    path = 'beeid_data/images/'
    meta_dict = {"identities": []}
    meta_list = []
    bee_in = 0
    for img in sorted(os.listdir(path)):
        bee_id, _, _ = img.split('_')
        print(bee_id)
        if int(bee_id) == bee_in:
            meta_list.append(img)
        else:
            meta_dict["identities"].append([meta_list])
            meta_list = [img]
            bee_in += 1
    with open('beeid_data/meta.json', 'w') as outfile:
        json.dump(meta_dict, outfile)


def generate_split_json(test_percentage=0.8, gallery_to_query_ratio=10):
    path = 'beeid_data/images/'
    last_id = int(sorted(os.listdir(path))[-1].split('_')[0])
    last_train_id = int(test_percentage * last_id+1)

    trainval = list(range(0, last_train_id))
    gallery = list(range(last_train_id, last_id+1))
    query = gallery[::gallery_to_query_ratio]

    split_dict = {"trainval": trainval,
                  "gallery": gallery,
                  "query": query}

    with open('beeid_data/splits.json', 'w') as outfile:
        json.dump(split_dict, outfile)


# delete unnecessary folders
def clean_up():
    print('Thanks for using the datamodulemaker')
    print('Once we cleaned up, you are ready to go..')
    print('Please cite this code in your work - do not hesitate to ask questions')
    print('Contact: mark.schutera@mailbox.org / mark.schutera@kit.edu')
    # delete folders
    try:
        for folder in filenames:
            shutil.rmtree(folder)
        # delete txt files
        os.remove('requirements.txt')
        os.remove('datamodulemaker.py')
        # move readme
        shutil.move('readme.txt', 'beeid_data/')
    except WindowsError:
        print()


# -------

if os.path.exists('beeid_data/images/') is False:
    extract_all_files(filenames)

    for name in filenames:
        # create raw frames from mp4 files {:04d}.format(count)
        mp4_to_frames(name)


generate_crops2(filenames)
generate_meta_json()
generate_split_json()

# remove old folders S1, S2, S3, S4, S5 to clean up.
# clean_up()
