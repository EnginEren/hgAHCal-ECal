import numpy as np
import h5py 
import argparse
import os


def merge_hdf5s(inptFList, outF):
    
    f_list = []
    for name in inptFList:
        f_list.append(h5py.File(name, 'r'))

    length_in_list = []
    for f in f_list:
        length_in_list.append(f['ecal']['layers'].shape[0])
        
        print ("ECAL: ")
        print(f['ecal']['layers'].shape)
        print(f['ecal']['energy'].shape)

        print ("HCAL: ")
        print(f['hcal']['layers'].shape)
        print(f['hcal']['energy'].shape)        

    
    
    f = h5py.File(outF,'w-')
    grp = f.create_group("ecal")
    dset = grp.create_dataset('layers', shape=(0, 30, 30, 30), maxshape=(None, 30, 30, 30), chunks=(100, 30, 30, 30), dtype='f8')
    dsetE = grp.create_dataset('energy', shape=(0, 1), maxshape=(None, 1), chunks=(100, 1), dtype='f4')
   
    hgrp = f.create_group("hcal")
    hcal_dset = hgrp.create_dataset('layers', shape=(0, 48, 30, 30), maxshape=(None, 48, 30, 30), chunks=(100, 48, 30, 30), dtype='f8')
    


    batchsize = 100

    for f_ind in range(len(f_list)):
        f_in_1 = f_list[f_ind]
        length_in_1 = length_in_list[f_ind]

        for index in np.arange(0,length_in_1,step=batchsize):
            
            ### ECAL 
            ds1 = f_in_1['ecal']['layers'][index:index+batchsize]
            dset.resize(dset.shape[0]+ds1.shape[0], axis=0)
            dset[-1*ds1.shape[0]:] = ds1

            ds1E = f_in_1['ecal']['energy'][index:index+batchsize]   
            
            dsetE.resize(dsetE.shape[0]+ds1E.shape[0], axis=0)
            dsetE[-1*ds1E.shape[0]:] = ds1E
        

            ###
            ### HCAL
            hds1 = f_in_1['hcal']['layers'][index:index+batchsize]
            hcal_dset.resize(hcal_dset.shape[0]+hds1.shape[0], axis=0)
            hcal_dset[-1*hds1.shape[0]:] = hds1

       
            
    f.close()


    print("Output file was created: ", outF)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
   

    parser.add_argument("--input", type=str, nargs="+", required=True, help='input hdf5 files')
    parser.add_argument('--output', type=str, required=True, help='merged hdf5 file')


    opt = parser.parse_args()

    out = str(opt.output)
    inputH5s = opt.input

    print (inputH5s)

    

    merge_hdf5s(inputH5s, out)

  