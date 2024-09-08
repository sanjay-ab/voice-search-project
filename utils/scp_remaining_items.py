"""Use to download or upload files from/to a server using SCP."""
import os
import paramiko
from scp import SCPClient
import wave

def check_wav_file(file_path):
    """Check if a WAV file is corrupted or not.

    Args:
        file_path (str): path to wav file
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # If we can open the file and read its parameters, it's likely not corrupted
            wav_file.getparams()
        # print(f"{file_path} is formatted correctly.")
    except wave.Error as e:
        print(f"{file_path} is corrupted or not a valid WAV file. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {file_path}. Error: {e}")

def get_files_list_from_transcript(transcript_file, language):
    """From a transcript file, get the list of files that are present in the dataset.

    Args:
        transcript_file (str): path of transcript file
        language (str): language of the dataset

    Returns:
        set{str}: set of all files in the dataset
    """
    if language in ["hindi", "marathi", "odia"]:
        split_char = " "
    elif language in ["gujarati", "telugu"]:
        split_char = "\t"

    files_list = []
    with open(transcript_file, "r", encoding="utf-8") as file:
        for line in file:
            file_name = line.split(split_char)[0]
            files_list.append(f"{file_name.strip()}.wav")
    return set(files_list)

def createSSHClient(hostname, user, password=None, key_filename=None):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if password is not None:
        client.connect(hostname=hostname, port=22, username=user, password=password)
    elif key_filename is not None:
        client.connect(hostname=hostname, port=22, username=user, key_filename=key_filename)
    else:
        raise ValueError("Either password or key_filename must be provided.")
    return client

def download_missing_files(dest_dir, language, missing_files):
    """Download missing files from the Edinburgh University Informatics server.

    Args:
        dest_dir (str): destination directory to save files
        language (str): language of the dataset
        missing_files (set{str}): set of missing files to download
    """
    with(open("password.txt", "r")) as file:
        password = file.read().strip()
    
    total_num_files = len(missing_files) 

    ssh = createSSHClient("student.ssh.inf.ed.ac.uk", "s2556413", password=password)
    with SCPClient(ssh.get_transport()) as scp:
        for i, file_name in enumerate(missing_files):
            print(f"Downloading file: {file_name}")
            scp.get(f"/group/corporapublic/ASR_challenge/{language}/train/audio/{file_name}", dest_dir)
            print(f"Percentage of files downloaded: {i/total_num_files*100:.2f}%")

def upload_missing_files(source_dir, language, existing_files):
    """Upload missing files to the Cirrus server.

    Args:
        source_dir (str): source directory to upload files from
        language (str): language of the dataset
        existing_files (set{str}): set of files that are already uploaded
    """
    files = os.listdir(source_dir)
    total_num_files= len(files) 

    ssh = createSSHClient("login.cirrus.ac.uk", "sanjayb", key_filename="C:/Users/sanja/.ssh/safe_key")
    with SCPClient(ssh.get_transport()) as scp:
        for i, file_name in enumerate(files):
            if file_name in existing_files:
                print(f"File {file_name} already uploaded.")
                continue
            print(f"Uploading file: {source_dir}/{file_name}")
            scp.put(f"{source_dir}/{file_name}", 
                    f"/work/tc062/tc062/sanjayb/voice-search-project/data/{language}/all_data/{file_name}")
            print(f"Percentage of files uploaded: {i/total_num_files*100:.2f}%")

def check_files(dest_dir, files_list):
    """Check if all files in the list are present in the destination directory 
    and are valid WAV files.

    Args:
        dest_dir (str): path of directory to check
        files_list (set{str}): set of all files that should be present in the directory
    """
    percentage_threshold = 0
    num_files = len(files_list)
    dir_files = os.listdir(dest_dir)

    for i, file_name in enumerate(files_list):
        if file_name not in dir_files:
            print(f"File {file_name} is missing.")
            continue
        file_path = os.path.join(dest_dir, file_name)
        check_wav_file(file_path)

        if i/num_files*100 > percentage_threshold:
            print(f"Percentage of files checked: {percentage_threshold:.2f}%")
            percentage_threshold += 10

    print("All files checked.")

def load_existing_files_from_file(file_path):
    existing_files = set()
    with open(file_path, "r") as file:
        for line in file:
            existing_files.add(line.strip())
    return existing_files

if __name__ == "__main__":

    language = "hindi_upsampled"
    behaviour = "upload"

    print(f"language: {language}")
    print(f"behaviour: {behaviour}")

    if behaviour == "download":
        dest_dir = f"{language}/train/audio"
        transcript_file = f"{language}/train/transcription.txt"
        files_list = get_files_list_from_transcript(transcript_file, language)
        existing_files = set(os.listdir(dest_dir))
        missing_files = files_list - existing_files
        download_missing_files(dest_dir, language, missing_files)

    if behaviour == "check":
        for directory in ["test", "train"]:
            print(f"Checking {directory} files.")
            dir_to_test = f"{language}/{directory}/audio"
            transcript_file = f"{language}/{directory}/transcription.txt"
            files_list = get_files_list_from_transcript(transcript_file, language)
            check_files(dir_to_test, files_list)

    if behaviour == "upload":
        existing_files = load_existing_files_from_file(f"{language}/uploaded_files.txt")
        for directory in ["test", "train"]:
            print(f"Uploading {directory} files.")
            upload_dir = f"{language}/{directory}/audio"
            upload_missing_files(upload_dir, language, existing_files)
