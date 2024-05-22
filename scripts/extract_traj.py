import os
import tqdm
import struct
import argparse

import numpy as np
import torch

def to_bytes(s,enc='utf-8'):
    return bytes(s+'\0',enc)

def write_raw(f, name, obj):
    f.write(to_bytes(name))
    if isinstance(obj,str):
        f.write(to_bytes('str'))
        f.write(to_bytes(obj))
    elif isinstance(obj,int):
        f.write(to_bytes('int'))
        f.write(struct.pack('q'*1,obj))
        #f.write(obj.to_bytes(8,'little'))
    elif isinstance(obj,float):
        f.write(to_bytes('float'))
        f.write(struct.pack('d'*1,obj))
    elif isinstance(obj,np.ndarray):
        f.write(to_bytes(str(obj.dtype)))
        n_shape = len(obj.shape)
        shape_ = [n_shape] + list(obj.shape)
        f.write(struct.pack('q' * (n_shape + 1), *shape_))
        obj.tofile(f)

def get_all_files_in_folder(srcfolder):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(srcfolder):
        f.extend(filenames)
        break
    return f

def extract_traj(src_folder, tar_folder):

    os.makedirs(tar_folder, exist_ok=True)

    file_names = get_all_files_in_folder(src_folder)
    file_names.sort()

    for fn in tqdm.tqdm(file_names):
        obj = torch.load(os.path.join(src_folder,fn))
        rf,sf = os.path.splitext(fn)

        exp = obj['flame']['exp']
        shape = obj['flame']['shape']
        tex = obj['flame']['tex']
        sh = obj['flame']['sh']
        eyes = obj['flame']['eyes']
        eyelids = obj['flame']['eyelids']
        jaw = obj['flame']['jaw']

        R = obj['opencv']['R']
        t = obj['opencv']['t']
        K = obj['opencv']['K']

        img_size = obj['img_size']
        frame_id = obj['frame_id']
        global_step = obj['global_step']

        f = open(os.path.join(tar_folder, rf + '.bin'), 'wb')
        write_raw(f, 'exp', exp)
        write_raw(f, 'shape', shape)
        write_raw(f, 'tex', tex)
        write_raw(f, 'sh', sh)
        write_raw(f, 'eyes', eyes)
        write_raw(f, 'eyelids', eyelids)
        write_raw(f, 'jaw', jaw)
        write_raw(f, 'R', R)
        write_raw(f, 't', t)
        write_raw(f, 'K', K)
        write_raw(f, 'img_size', img_size)
        write_raw(f, 'frame_id', frame_id)
        write_raw(f, 'global_step', global_step)
        f.close()

        # id = int(rf)
        # if id % 10 == 0:
        #     print(id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transform format')
    parser.add_argument('-i','--input',type=str,help='input folder')
    parser.add_argument('-o','--output',type=str,help='output folder')
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(unknown)
        exit(-1)
    extract_traj(args.input, args.output)