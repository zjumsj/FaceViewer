import numpy as np
import os
import pickle
import argparse

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def print(self):
        L = [a for a in dir(self) if not a.startswith('__')]
        for a in L:
            if a in ['print','dump']:
                continue
            print(a)

    def dump(self, tar_folder):
        L = [a for a in dir(self) if not a.startswith('__')]
        for a in L:
            if a in ['print', 'dump']:
                continue
            obj = getattr(self,a)
            if isinstance(obj,str):
                f = open(os.path.join(tar_folder, a+'.txt'), "w")
                f.write(obj)
                f.close()
            else:
                obj = np.asarray(obj)
                np.save(os.path.join(tar_folder, a + '.npy'), obj)
                #elif isinstance(obj,np.ndarray):
                #    np.save(os.path.join(tar_folder,a+'.npy'), obj)

def extract_tensor_from_FLAME(srcfile, tar_folder):

    os.makedirs(tar_folder, exist_ok=True)
    with open(srcfile, 'rb') as f:
        # flame_model = Struct(**pickle.load(f, encoding='latin1'))
        ss = pickle.load(f, encoding='latin1')
        flame_model = Struct(**ss)

    #flame_model.J_regressor = flame_model.J_regressor.toarray()
    flame_model.J_regressor = flame_model.J_regressor.todense(order='C')
    flame_model.kintree_table = flame_model.kintree_table.astype(np.int32)

    flame_model.print()
    flame_model.dump(tar_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract tensor from FLAME2020 pkl')
    parser.add_argument('-i', '--input', type=str, help='input filename')
    parser.add_argument('-o', '--output', type=str, help='output folder')
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(unknown)
        exit(-1)
    extract_tensor_from_FLAME(args.input, args.output)
