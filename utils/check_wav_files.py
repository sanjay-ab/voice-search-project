"""Use to check if all files in a directory are valid WAV files."""
import sys
import os
import wave

from utils.scp_remaining_items import check_wav_file, check_files

if __name__ == "__main__":
    dest_dir = sys.argv[1]
    files_list = os.listdir(dest_dir)
    check_files(dest_dir, files_list)