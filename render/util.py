import json 
import os
import pathlib

def create_tmp_dir():
    directory = (pathlib.Path('.') / 'tmp').resolve()
    if not directory.is_dir():
        os.mkdir(directory)
    return directory


def write_filelist_json(files):

    tmp_dir = create_tmp_dir()

    dict = {"files": files}

    filepath = tmp_dir / "filelist.json"
    with open(filepath, "w") as outfile:
        json.dump(dict, outfile)
