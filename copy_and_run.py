import os.path
import uuid
from datetime import datetime
import shutil

from constants import FOLDERS_TO_EXCLUDE_COPYING


def copy_and_run(src, dest, files_or_folders_to_exclude):
    unique_id = str(uuid.uuid4()).split('-')[0]
    time_stamp = datetime.now().strftime("%d-%b-%Y__%H:%M:%S")
    folder_to_copy = os.path.join(dest, os.path.basename(src) + '_' + time_stamp + '_' + unique_id)

    def _ignore_(path, names):
        return set(files_or_folders_to_exclude)

    shutil.copytree(src, folder_to_copy, ignore=_ignore_)
    print(f'Copied repo to: {folder_to_copy}')
    return folder_to_copy


def default_copy_and_run(dest):
    repo_path = copy_and_run(
        src=os.path.dirname(os.path.realpath(__file__)),
        dest=dest,

        files_or_folders_to_exclude=FOLDERS_TO_EXCLUDE_COPYING)
    return os.path.join(repo_path, 'main.py')

if __name__ == '__main__':
    default_copy_and_run()