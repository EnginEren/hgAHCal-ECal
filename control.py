from pyLCIO import EVENT, UTIL, IOIMPL, IMPL, IO
#import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py 
import awkward1 as ak 
import argparse
import math
import pickle
from functions import CellIDDecoder
from functions import plt_ExampleImage
import matplotlib.pyplot as plt
import random
import array as arr


font = {'family' : 'serif',
        'size'   : 18}
mpl.rc('font', **font)
plt.style.use('classic')
mpl.rc('font', **font)

hmap = np.array([1806.5, 1810.5, 1815.5, 1819.5, 1823.5, 1828.0, 1832.5, 1836.5, 1841.5, 
                   1845.5, 1849.5, 1853.5, 1858.5, 1862.5, 1867.5, 1871.5, 1875.5, 1879.5, 
                   1884.5, 1888.5, 1893.5, 1899.5, 1903.5, 1910.5, 1914.5, 1920.5, 1925.5, 
                   1931.5, 1936.5, 1942.5, 
                   ### HCAL
                   2079, 2108, 2134, 2160, 2187, 2213, 2239, 2265, 2294, 2320, 2346,
                   2372, 2398, 2424, 2450, 2479, 2506, 2532, 2558, 2584, 2610, 2636,
                   2662, 2691, 2717, 2743, 2769, 2796, 2822, 2848, 2877, 2903, 2929,
                   2955, 2981, 3007, 3033, 3062, 3088, 3115, 3141, 3167, 3193, 3219,
                   3248, 3274, 3300, 3326])


def fill_record(inpLCIO, colEcal, colHcal, nevents):
    """this function reads all events in LCIO file and put them into awkward array"""

    ## open LCIO file
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( inpLCIO )

    ## create awkward array
    b = ak.ArrayBuilder()

    ## start looping
    nEvt = 0
    for evt in reader:
        nEvt += 1
        if nEvt > nevents:
            break

        
        ## First thing: MC particle collection
        b.begin_list()
        mcparticle = evt.getCollection("MCParticle")    
        
        
        ## fill energy for each MCParticle (in this case just an incoming photon)
        for enr in mcparticle:
            b.begin_record()
            b.field("E")
            b.real(enr.getEnergy())
            ## calculate polar angle theta and fill
            b.field("theta")
            pVec = enr.getMomentum()
            theta = math.pi/2.00 - math.atan(pVec[2]/pVec[1])
            b.real(theta)
            b.end_record() 
    
        ## ECAL barrel collection
        ecalBarrel = evt.getCollection(colEcal)
        cellIDString = ecalBarrel.getParameters().getStringVal("CellIDEncoding")
        decoder = CellIDDecoder( cellIDString ) 
        ##

        for hit in ecalBarrel:

            l = decoder.layer( hit.getCellID0() ) ## get the layer information from CellID0 
            e = hit.getEnergy() 
            pos = hit.getPosition()
        
            
            ## start filling a record with all relevant information
            b.begin_record() 
            b.field("x")
            b.real(pos[0])
            b.field("y")
            b.real(pos[1])
            b.field("z")
            b.real(pos[2])
            b.field("e")
            b.real(e * 1000)
            b.field("layer")
            b.integer(l)
            b.field("cid0")
            b.integer(hit.getCellID0())
            b.field("cid1")
            b.integer(hit.getCellID1())
            b.end_record() 

        
        ## HCAL barrel collection
        hcalBarrel = evt.getCollection(colHcal)
        cellIDString = hcalBarrel.getParameters().getStringVal("CellIDEncoding")
        hcal_decoder = CellIDDecoder( cellIDString ) 
        ##

        for hit in hcalBarrel:

            l = hcal_decoder.layer( hit.getCellID0() ) ## get the layer information from CellID0 
            e = hit.getEnergy() 
            pos = hit.getPosition()

            ## start filling a record with all relevant information
            b.begin_record() 
            b.field("x")
            b.real(pos[0])
            b.field("y")
            b.real(pos[1])
            b.field("z")
            b.real(pos[2])
            b.field("e")
            b.real(e * 1000)
            b.field("layer")
            b.integer(30+l)
            b.field("cid0")
            b.integer(hit.getCellID0())
            b.field("cid1")
            b.integer(hit.getCellID1())
            b.end_record() 
        
        b.end_list()

    ### Example:
    # Get the incident energy of the first event --> b[0].E
    # Get the x positions of the the first event --> b[0].x 

    return b


def plot_hits(n, rd):
    
    z = ak.to_numpy(rd[n].z)
    y = ak.to_numpy(rd[n].y)
    x = ak.to_numpy(rd[n].x)

    ### X-Z projection
    figXZ, axXZ = plt.subplots(figsize=(10, 10))
    axXZ.plot(x.data, z.data, 'o', c='blue', label='Geant4')

    axXZ.set(xlabel='x [mm]', ylabel='z [mm]',
        title='')
    axXZ.grid()

    plt.xlim(-700,700)
    plt.ylim(-100,2100)
    figXZ.savefig("/mnt/plots/event"+str(n)+"__z-x.png")
   

    ### X-Y projection
    figXY, axXY = plt.subplots(figsize=(10, 10))
    axXY.plot(x.data, y.data, 'o', c='blue', label='Geant4')

    axXY.set(xlabel='x [mm]', ylabel='y [mm]',
        title='')
    axXY.grid()
    axXY.legend()

    plt.xlim(-150,100)
    plt.ylim(1700,3500)
    figXY.savefig("/mnt/plots/event"+str(n)+"__y-x.png")

    ### Z-Y projection
    figZY, axZY = plt.subplots(figsize=(10, 10))
    axZY.plot(z.data, y.data, 'o', c='blue', label='Geant4')

    axZY.set(xlabel='z [mm]', ylabel='y [mm]',
        title='')
    axZY.grid()
    axZY.legend()

    plt.xlim(990, 1100)
    plt.ylim(1700,3500)
    figZY.savefig("/mnt/plots/event"+str(n)+"__z-y.png")






if __name__=="__main__":

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--lcio', type=str, required=True, help='input LCIO file')
    parser.add_argument('--nEvents', type=int, help='number of events', default=10)

    opt = parser.parse_args()

    nEvents = int(opt.nEvents)
    lcioFile = str(opt.lcio)

    record = fill_record(lcioFile, "EcalBarrelCollection", "HcalBarrelRegCollection", nEvents)  
    for i in [1,5,10]:
        plot_hits(i, record)