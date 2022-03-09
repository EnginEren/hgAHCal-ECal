
import numpy as np
import h5py 
import argparse
import math
import os


def merge_hdf5s(inptFList, outF):
    with h5py.File(outF, mode='w') as h5fw:
        for h5name in inptFList:
            print ("Reading the file: ", h5name)
            h5fr = h5py.File(h5name,'r') 
            for obj in h5fr.keys():        
                h5fr.copy(obj, h5fw)       
    
    print("Output file was created: ", outF)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
   

    parser.add_argument("--input", type=str, nargs="+", required=True, help='input hdf5 files')
    parser.add_argument('--output', type=str, required=True, help='merged hdf5 file')


    opt = parser.parse_args()

    out = str(opt.output)
    inputH5s = str(opt.input)

    
    realList = [ ]
    for i in inputH5s:
        realList.append(os.popen('cat {}'.format(i)).read().rstrip("\n"))


    merge_hdf5s(realList, out)

  