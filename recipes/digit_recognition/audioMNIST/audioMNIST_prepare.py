import os
import shutil
import sys
import random
import logging
from speechbrain.utils.data_utils import get_all_files, download_file

logger = logging.getLogger(__name__)
audioMNIST_GITHUB_URL = 'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip'

DATA_CSV = 'data.csv'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
TEST_CSV = 'test.csv'


def prepare_audioMNIST(
        data_folder,
        save_folder,
        split_ratio=[4, 1, 1],
        split_base='id',
        reprepare=False
):
    """
    Prepares csv files for audioMNIST dataset

    Downloads the dataset if it is not found in the `data_folder`
    source: https://github.com/Jakobovski/free-spoken-digit-dataset

    Arguments
    ----------
    data_folder : str
        Path to the folder where the audioMNIST dataset is stored.
    save_folder : str
        Path where data specification files will be saved including:
            data.csv: specification file for all data
            train.csv: specification file for training data
            val.csv: specification file for validation data
            test.csv: specification file for testing data
    split_ratio : [int, int, int]
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. AudioMNIST will be split by speakers based on the ratio
    reprepare: boolean
        If data have already been prepared, whether or not data is required to be prepared again

     Example
    -------
    >>> data_folder = '/path/to/audioMNIST'
    >>> save_folder = '/path/to/saved_csv'
    >>> prepare_mini_librispeech(data_folder, save_folder)
    """
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_val = os.path.join(save_folder, VAL_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)
    if skip(save_csv_train, save_csv_val, save_csv_test) and not reprepare:
        logger.info("Preparation completed in previous run, skipping.")
    else:
        logger.info("Data_preparation...")
    if split_base != 'speaker' and split_base != 'id':
        raise ValueError('Split base should be either speaker or id')
    if not check_folders(os.path.join(data_folder, 'free-spoken-digit-dataset-master')):
        logger.info(f"Downloading {audioMNIST_GITHUB_URL} to {data_folder}")
        download_audioMNIST(data_folder)
        logger.info("Downloading finished")

    else:
        logger.info("Dataset already downloaded.")
    logger.info(
        f"Creating {DATA_CSV}, {TRAIN_CSV}, {VAL_CSV}, and {TEST_CSV} under {save_folder}"
    )
    num_audios, digit_list, speaker_list, saved_data_path = _create_data_csv(data_folder, save_folder, ['.wav'])
    if split_base == 'speaker':
        _split_sets_by_speaker(saved_data_path, save_folder, speaker_list, split_ratio)
    else:
        _split_sets_by_id(saved_data_path, save_folder, num_audios, split_ratio)
    logger.info(
        f"Successfully created: {DATA_CSV}, {TRAIN_CSV}, {VAL_CSV} and {TEST_CSV}"
    )


def _create_data_csv(audio_folder, save_folder, extensions):
    """
    Creating data.csv including all data in the dataset

    Arguments
    ----------
    audio_folder: str
        path to the folder where the dataset is downloaded
    save_folder: str
        Path to the folder where data.csv will be saved
    extensions: list
        extensions of audio files
    """
    audios_path = os.path.join(audio_folder + '/', 'free-spoken-digit-dataset-master/', 'recordings')
    files = get_all_files(audios_path, match_and=extensions)
    digit_list = []
    speaker_list = []
    audio_id = 0
    f = open(os.path.join(save_folder, 'data.csv'), 'w')
    f.write('ID, file_path, digit_label, speaker_name, index\n')
    for file in files:
        file = file.replace('\\', '/')
        audio_file_name = file.split('/')[-1]
        for extension in extensions:
            audio_file_name = audio_file_name.replace(extension, '')
        digit_label, speaker_name, index = audio_file_name.split('_')
        # print(digit_label, speaker_name, index)
        if digit_label not in digit_list:
            digit_list.append(digit_label)
        if speaker_name not in speaker_list:
            speaker_list.append(speaker_name)
        f.write('%d, %s, %s, %s, %s\n' % (audio_id, file, digit_label, speaker_name, index))
        audio_id = audio_id + 1
    f.close()
    return audio_id, digit_list, speaker_list, os.path.join(save_folder, 'data.csv')


def _split_sets_by_speaker(saved_data_path, save_folder, speaker_list, split_ratio):
    """
    Splitting data.csv into train.csv, val.csv and test.csv based on split ratio
    The splitting is based on speakers.

    Arguments
    ----------
    saved_data_path: str
        Path where data.csv is saved
    save_folder: str
        Path to the folder where train.csv, val.csv, test.csv will be saved
    speaker_list: list
        a list of all speakers in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
            and test sets
    """
    speaker_num = len(speaker_list)
    test_split_num = int(speaker_num * split_ratio[2] / sum(split_ratio))
    if test_split_num < 1:
        logger.info('Too less test speakers, updating the number of test speakers to 1')
        test_split_num = 1
    val_split_num = int(speaker_num * split_ratio[1] / sum(split_ratio))
    if val_split_num < 1:
        logger.info('Too less validation speakers, updating the number of test speakers to 1')
        val_split_num = 1
    train_split_num = speaker_num - val_split_num - test_split_num
    random.shuffle(speaker_list)
    f_train = open(os.path.join(save_folder, 'train.csv'), 'w')
    f_val = open(os.path.join(save_folder, 'val.csv'), 'w')
    f_test = open(os.path.join(save_folder, 'test.csv'), 'w')
    f = open(saved_data_path, 'r')
    # write header
    line = f.readline()
    f_train.write(line)
    f_val.write(line)
    f_test.write(line)
    while line is not None:
        line = f.readline()
        if not line:
            break
        speaker_name = line.split(',')[-2].strip()
        # print(speaker_name)
        speaker_idx = speaker_list.index(speaker_name)
        if speaker_idx < train_split_num:
            f_train.write(line)
        elif train_split_num <= speaker_idx < train_split_num + val_split_num:
            f_val.write(line)
        else:
            f_test.write(line)
    f_train.close()
    f_val.close()
    f_test.close()


def _split_sets_by_id(saved_data_path, save_folder, num_audios, split_ratio):
    test_split_num = int(num_audios * split_ratio[2] / sum(split_ratio))
    val_split_num = int(num_audios * split_ratio[1] / sum(split_ratio))
    train_split_num = num_audios - test_split_num - val_split_num
    id_list = [i for i in range(num_audios)]
    random.shuffle(id_list)
    f_train = open(os.path.join(save_folder, 'train.csv'), 'w')
    f_val = open(os.path.join(save_folder, 'val.csv'), 'w')
    f_test = open(os.path.join(save_folder, 'test.csv'), 'w')
    f = open(saved_data_path, 'r')
    # write header
    line = f.readline()
    f_train.write(line)
    f_val.write(line)
    f_test.write(line)
    while line is not None:
        line = f.readline()
        if not line:
            break
        audio_id = int(line.split(',')[0].strip())
        # print(speaker_name)
        audio_id_idx = id_list.index(audio_id)
        if audio_id_idx < train_split_num:
            f_train.write(line)
        elif train_split_num <= audio_id_idx < train_split_num + val_split_num:
            f_val.write(line)
        else:
            f_test.write(line)
    f_train.close()
    f_val.close()
    f_test.close()


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def download_audioMNIST(
        destination
):
    """Download audioMNST dataset from github and unpack it

    Arguments
    ----------
     destination: str
        place to put dataset
    """
    audioMNIST_archive = os.path.join(destination, "audioMNIST.zip")
    download_file(audioMNIST_GITHUB_URL, audioMNIST_archive)
    shutil.unpack_archive(audioMNIST_archive, destination)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    data_folder = "D:/Documents/summer_project/audioMNIST"
    prepare_audioMNIST(data_folder, data_folder, cover_prev=True)
