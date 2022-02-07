from pyLCIO import EVENT, UTIL, IOIMPL, IMPL, IO
import matplotlib as mpl
import numpy as np
import h5py 
import awkward1 as ak 
import argparse
import math
from functions import CellIDDecoder
import matplotlib.pyplot as plt
import array as arr
import os

font = {'family' : 'serif',
        'size'   : 18}
mpl.rc('font', **font)
plt.style.use('classic')
mpl.rc('font', **font)

hmap = np.array([   1811, 1814, 1824, 1827, 1836, 1839, 1849,
                    1852, 1861, 1864, 1873, 1877, 1886, 1889, 1898, 1902,
                    1911, 1914, 1923, 1926, 1938, 1943, 1955, 1960,
                    1971, 1976, 1988, 1993, 2005, 2010, 
                ### HCAL starts
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
    

        ## PFO collection
        pfoCand = evt.getCollection("PandoraPFOs")
        for p in pfoCand:
            b.begin_record()
            b.field("pfoE")
            b.real(p.getEnergy())
            b.field("pfoCh")
            b.real(p.getCharge())
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



    
def fill_numpyECAL(record, nEvents):
    """this function reads the awkward array and edits for our needs: Projecting into 30x30 grid"""
    
    #defined binning
    binX = np.arange(-235, -60, 5.088333)
    #binZ large 
    binZ = np.arange(68,222, 5.088333)

    ## Unable to escape using python list here. But we can live with that.
    l = []
    E = []
    
    for i in range(0, nEvents):

        #Get hits and convert them into numpy array 
        z = ak.to_numpy(record[i].z)
        x = ak.to_numpy(record[i].x)
        y = ak.to_numpy(record[i].y)
        e = ak.to_numpy(record[i].e)

        #Get indicent energies
        incE = ak.to_numpy(record[i].E)
        E.append(incE.compressed()[0])



        layers = []
        #loop over layers and project them into 2d grid.
        for j in range(0,30):
            idx = np.where((y <= (hmap[j] + 0.9999)) & (y > (hmap[j] + 0.0001)))
            xlayer = x.take(idx)[0]
            zlayer = z.take(idx)[0]
            elayer = e.take(idx)[0]
            H, xedges, yedges = np.histogram2d(xlayer, zlayer, bins=(binX, binZ), weights=elayer)
            layers.append(H)

   
        l.append(layers)
    
    ## convert them into numpy array
    shower = np.asarray(l)
    e0 = np.reshape(np.asarray(E), (-1,1))
    

    return shower, e0






if __name__=="__main__":

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--lcio', type=str, required=True, help='input LCIO file')
    parser.add_argument('--nEvents', type=int, help='number of events', default=10)
    parser.add_argument('--outputR', type=str, required=True, help='run name of output hdf5 file')
    parser.add_argument('--outputP', type=str, required=True, help='part name of output hdf5 file')

    opt = parser.parse_args()

    nEvents = int(opt.nEvents)
    lcioFile = str(opt.lcio)
    outP = str(opt.outputP)
    outR = str(opt.outputR)

    record = fill_record(lcioFile, "EcalBarrelCollection", "HcalBarrelRegCollection", nEvents)  
    showers, e0 = fill_numpyECAL(record, nEvents)

    #Open HDF5 file for writing
    hf = h5py.File('/mnt/run_' + outR + '/pion-shower_' + outP + '.hdf5', 'w')
    grp = hf.create_group("ecal")

    ## write to hdf5 files
    grp.create_dataset('energy', data=e0)
    grp.create_dataset('layers', data=showers)

    #close file
    hf.close()
    