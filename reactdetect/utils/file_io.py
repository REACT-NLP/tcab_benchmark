import joblib
import os

def grab_joblibs(root_dir):
    """
    grab all joblibs in a root dir
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(root_dir):
        for file in f:
            if 'joblib' in file:
                files.append(os.path.join(r, file))
    return files

import  sys
def path_splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def grab_fpath_with(root_dir, string):
    """
    grab all joblibs in a root dir
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(root_dir):
        for file in f:
            if string in file:
                files.append(os.path.join(r, file))
    return files


def mkfile_if_dne(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        print('mkfile warning: making fdir since DNE: ',fpath)
        os.makedirs(os.path.dirname(fpath))
    else:
        pass

def mkdir_if_dne(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)

from shutil import rmtree
def rm_rf_dir(dpath):    
    rmtree(dpath)

def vim_write_zz(fpath, string):
    """
    write as if vim write zz
    """
    mkfile_if_dne(fpath)
    with open(fpath,'w') as ofp:
        ofp.write(string+'\n')

def vim_append_zz(fpath, string):
    """
    write as if vim write zz
    """
    mkfile_if_dne(fpath)
    with open(fpath,'a') as ofp:
        ofp.write('\n')
        ofp.write(string+'\n')



import json
def load_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data

def dump_json(data, fpath):
    with open(fpath, 'w') as outfile:
        json.dump(data, outfile)