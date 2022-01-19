from ROOT import TH1D, TCanvas, TF1, std
from pyLCIO import EVENT, UTIL, IOIMPL, IMPL
import matplotlib.pyplot as plt
import numpy as np


import matplotlib as mpl



import awkward1 as ak

import math
import sys
import string

color_list = []
color_list.append('dimgrey')
color_list.append('black')
color_list.append('green')

linewidth_list = []
linewidth_list.append(1.5)
linewidth_list.append(2)
linewidth_list.append(2)

    
linestyle_list = []
linestyle_list.append('-')
linestyle_list.append('--')
linestyle_list.append('--')

fillcolor_list = []
fillcolor_list.append('lightgrey')
fillcolor_list.append('red')
fillcolor_list.append('violent')

def Average(lst):
    return sum(lst) / len(lst)



## returns fractional energy: first 20 layers and last 10 layers
def fracEsum(fName, maxEvt, collection):
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( fName )
    
    esum20l = []
    esum10l = []

    
    nEvt = 0

    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
                break
        ecalBarrel = evt.getCollection(collection)
        cellIDString = ecalBarrel.getParameters().getStringVal("CellIDEncoding")
        decoder = CellIDDecoder( cellIDString ) 
        esum20 = 0.0
        esum10 = 0.0 
        for hit in ecalBarrel:
            l = decoder.layer( hit.getCellID0() ) 
            e = hit.getEnergy() 
            #print ("Energy:", hit.getEnergy(), " Cell ID0:", hit.getCellID0(), " layer: ", decoder.layer( hit.getCellID0() )) 
            if l < 20:
                esum20 += e 
            elif l >= 20:
                esum10 = esum10 + e*2
        
        esum20l.append(esum20)
        esum10l.append(esum10)
        
    esum20np = np.asarray(esum20l)
    esum10np = np.asarray(esum10l)
    
    return esum20np, esum10np
    

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def interval_quantile_(x, quant=0.9):
    """Calculate the shortest interval that contains the desired quantile"""
    x = np.sort(x)
    # the number of possible starting points
    n_low = int(len(x) * (1 - quant))
    # the number of events contained in the quantil
    n_quant = len(x) - n_low

    # Calculate all distances in one go
    distances = x[-n_low:] - x[:n_low]
    i_start = np.argmin(distances)

    return i_start, i_start + n_quant

def fit90(x):     
    x = np.sort(x)

    n10percent = int(round(len(x)*0.1))
    n90percent = len(x) - n10percent
    
    start, end = interval_quantile_(x, quant=0.9)
    
    rms90 = np.std(x[start:end])
    mean90 = np.mean(x[start:end])
    mean90_err = rms90/np.sqrt(n90percent)
    rms90_err = rms90/np.sqrt(2*n90percent)   # estimator in root
    return mean90, rms90, mean90_err, rms90_err

def Esumhit(fName, maxEvt, collection):
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( fName )
    
    esuml = []

    nEvt = 0

    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
                break
        ecalBarrel = evt.getCollection(collection)
    
        cellIDString = ecalBarrel.getParameters().getStringVal("CellIDEncoding")
        decoder = CellIDDecoder( cellIDString ) 
        esum = 0.0
        for hit in ecalBarrel:
            l = decoder.layer( hit.getCellID0() ) 
            e = hit.getEnergy() 
            esum += e
        
        esuml.append(esum)
        
        
    esumnp = np.asarray(esuml)
    
    
    return esumnp




def Esumhit(fName, maxEvt, collection):
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( fName )
    
    esuml = []
    nEvt = 0

    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
                break
        ecalBarrel = evt.getCollection(collection)
        esum = 0.0
        for hit in ecalBarrel:
            e = hit.getEnergy() 
            esum += e
        
        esuml.append(esum)
        
        
    esumnp = np.asarray(esuml)
    
    
    return esumnp





def pfo_cluster(fName, maxEvt, collection):
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( fName )
    
    esuml = []
    nhitl = []
    nEvt = 0

    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
                break
        cluster = evt.getCollection(collection)
        print (nEvt, cluster.getEnergy())
        
        
        chits = 0
        for hit in cluster:
            cal = hit.getCalorimeterHits() 
            esum = 0.0
            for h in cal:
                e = h.getEnergy()
                esum += e
                chits += 1
        
            esuml.append(esum)
            nhitl.append(chits)
        
        
    esumnp = np.asarray(esuml)
    nhitnp = np.asarray(nhitl)
    
    
    return esumnp, nhitnp


def pfo_Econt(fName, maxEvt, collection, trueE):

    reader = IOIMPL.LCFactory.getInstance().createLCReader()
   
    reader.open( fName )

    
    nEvt = 0
    LPFOfr = []
    sPFOfr = []
    PFO3fr = []
    PFO4fr = []
    
    
    b = ak.ArrayBuilder()
    w = ak.ArrayBuilder()
    
    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
                break
        
        col = evt.getCollection(collection)
        nPFOc = 0
        
        evtPFOs = []
        nPFO = []
        for c in col:
            e = c.getEnergy()  ## get energy
            evtPFOs.append(e)
            nPFOc += 1
        
        b.begin_list()
        w.begin_list()
        evtPFOs.sort(reverse=True)
        if nPFOc <= 4: 
            for i in range(1,nPFOc+1):
                b.integer(i)
                w.real( evtPFOs[i-1] / trueE) 
                #w.real ( evtPFOs[i-1] / sum(evtPFOs) )
        
        b.end_list()
        w.end_list()    
     
         
    
    
    return w, b


    
    
def getNpfos(fName, maxEvt, collection):

    reader = IOIMPL.LCFactory.getInstance().createLCReader()
   
    reader.open( fName )

    nEvt = 0
    l = []
    chL = []
    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
                break
        col = evt.getCollection(collection)
        nPFO = 0
        nCh = 0
        for c in col:
            nPFO+=1
            e = c.getEnergy()  ## get energy
            charge = c.getCharge()
            m = c.getMass()
            
        
        l.append(nPFO)
        
    
    return np.asarray(l)
    
    
def plt_g4(data_real, energy_center, maxE, minE, bins, xtitle,save_title, ymax, stats=True):
    figSE = plt.figure(figsize=(8,8))
    axSE = figSE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
    
   
    pSEa = axSE.hist(data_real, bins=bins, range=[minE, maxE], density=None, edgecolor=color_list[0],  
                   weights=np.ones_like(data_real)/(float(len(data_real))),
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')

  
    
    axSE.set_ylabel('normalized', family='serif')
    axSE.set_xlabel(xtitle, family='serif')
    axSE.set_xlim([minE, maxE])
    axSE.set_ylim([0, ymax])

  

    
    mu90, rms90, emu90, erms90 = fit90(data_real)
   
    
    
    posX = 0.02
    
    if stats:   
    
        axSE.text(posX, 0.65, "Geant4:", horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[0])
        axSE.text(posX, 0.60, "$\mu$ = {:.2f} $\pm$ {:.2f} ".format(mu90, emu90), horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[0],
                  fontsize=15)
        axSE.text(posX, 0.58, "$\sigma$ = {:.2f} $\pm$ {:.2f}".format(rms90, erms90), horizontalalignment='left',verticalalignment='top', 
                transform=axSE.transAxes, color = color_list[0],
                  fontsize=15)
    
    axSE.text(posX,
            0.97,
            '{:d} GeV Pions'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes)

   
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
    figSE.patch.set_facecolor('white')
    plt.savefig(save_title+ str(energy_center)+"_single_E_comp_dpi300.png", dpi=300)
        
       
    

def nhits(fName, maxEvt, collection):
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( fName )
    
    esuml = []

    nEvt = 0
    nhits = []
    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
            break
        
        col = evt.getCollection(collection)
        
        for c in col:
            hits = c.getCalorimeterHits().size()
        nhits.append(hits)    
     
    nphits = np.asarray(nhits)
    return nphits

def nhits_sim(fName, maxEvt, collection):
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( fName )
    
    nhits = []

    nEvt = 0
    #cut = 0.1e-03
    cut = 0.0
    for evt in reader:
        nEvt += 1
        if nEvt > maxEvt:
                break
        ecalBarrel = evt.getCollection(collection)
    
        cellIDString = ecalBarrel.getParameters().getStringVal("CellIDEncoding")
        decoder = CellIDDecoder( cellIDString ) 
        hits = 0
        for hit in ecalBarrel:
            if (hit.getEnergy() > cut):  
                hits += 1 
         
        
        nhits.append(hits)
        
        
    nsumnp = np.asarray(nhits)
    
    
    return nsumnp

def plot_linearity(x_data, data, data_err):
    figStr = plt.figure(figsize=(10,10))
    axStr = figStr.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
    markersize1=10
    markersize2=10
    
    pStra = axStr.errorbar(x=x_data, y=data, xerr=None, yerr=data_err, 
                               fmt='o', markersize=15, ecolor='lightgray', color='black')
    
    axStr.text(0.50, 0.87, 'BAE', horizontalalignment='left',verticalalignment='top', 
             transform=axStr.transAxes, color = 'green')
        

def plt_singleEnergy(data_real, data_fake, data_fake2, energy_center, maxE, minE, bins, xtitle,save_title, ymax, stats=True):
    figSE = plt.figure(figsize=(8,8))
    axSE = figSE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
    
   
    pSEa = axSE.hist(data_real, bins=bins, range=[minE, maxE], density=None, edgecolor=color_list[0],  
                   weights=np.ones_like(data_real)/(float(len(data_real))),
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')

    #axSE.plot((0.42, 0.48),(0.84, 0.84),linewidth=linewidth_list[0], 
    #         linestyle=linestyle_list[0], transform=axSE.transAxes, color = color_list[0]) 
    #axSE.text(0.50, 0.87, 'GEANT 4', horizontalalignment='left',verticalalignment='top', 
    #         transform=axSE.transAxes, color = color_list[0])


    
    pSpnEb = axSE.hist(data_fake, bins=pSEa[1], range=None, density=None, edgecolor=color_list[1], 
                   weights=np.ones_like(data_fake)/(float(len(data_fake))),    
                   label = "orig", linewidth=linewidth_list[1], linestyle=linestyle_list[1],
                   histtype='step')

    
    pSpnEc = axSE.hist(data_fake2, bins=pSEa[1], range=None, density=None,  edgecolor=color_list[2], 
                   weights=np.ones_like(data_fake2)/(float(len(data_fake2))),
                   label = "orig", linewidth=linewidth_list[2], linestyle=linestyle_list[2],
                   histtype='step')
    
    

    axSE.set_ylabel('normalized', family='serif')
    axSE.set_xlabel(xtitle, family='serif')
    axSE.set_xlim([minE, maxE])
    axSE.set_ylim([0, ymax])

  

   
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    #p0 = [10., 40., 8.]

    #bin_centres = (pSEa[1][:-1] + pSEa[1][1:])/2
    #bin_centres_wgan = (pSpnEb[1][:-1] + pSpnEb[1][1:]) / 2 
    
    
    #coeff, var_matrix = curve_fit(gauss, bin_centres, pSEa[0],p0=p0)
    #coeff_wgan, var_matrix = curve_fit(gauss, bin_centres_wgan, pSpnEb[0],p0=p0)
    
    # Get the fitted curve
    #hist_fit = gauss(bin_centres, *coeff)
    #hist_fit_wgan = gauss(bin_centres_wgan, *coeff_wgan)
    
    mu90, rms90, emu90, erms90 = fit90(data_real)
    wmu90, wrms90, ewmu90, ewrms90 = fit90(data_fake)
    bmu90, brms90, ebmu90, ebrms90 = fit90(data_fake2)
    
    
    #plt.plot(bin_centres, hist_fit, color='grey')
    #plt.plot(bin_centres_wgan, hist_fit_wgan, color='black')
    posX = 0.02
    
    if stats:   
    
        axSE.text(posX, 0.65, "Geant4:", horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[0])
        axSE.text(posX, 0.60, "$\mu$ = {:.2f} $\pm$ {:.2f}, $\sigma$ = {:.2f} $\pm$ {:.2f}".format(mu90, emu90, rms90, erms90), horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[0])


        axSE.text(posX, 0.55, "WGAN", horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[1])
        axSE.text(posX, 0.50, "$\mu$ = {:.2f} $\pm$ {:.2f}, $\sigma$ = {:.2f} $\pm$ {:.2f}".format(wmu90, ewmu90, wrms90, ewrms90 ),horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[1])


        axSE.text(posX, 0.75, "BIB-AE", horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[2])
        axSE.text(posX, 0.70, "$\mu$ = {:.2f} $\pm$ {:.2f}, $\sigma$ = {:.2f} $\pm$ {:.2f}".format(bmu90, ebmu90, brms90, ebrms90 ), horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[2])
    
    axSE.text(posX,
            0.97,
            '{:d} GeV Pions'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes)

   
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
    #plt.figtext(1.0, 0.2, data_real.describe())
    #plt.yscale('log')
    figSE.patch.set_facecolor('white')

    #hep.cms.label(loc=0)
    plt.savefig(save_title+ str(energy_center)+"_single_E_comp.png")

    
def plt_ExampleImage(image, model_title='ML Model', save_title='ML_model', mode='comp', draw_3D=False, n=1):

    #print(image.shape)
    #data[B,Z,X,Y]
    
    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)
    
    
    #image = tf_logscale_rev_np(np.reshape(image,(1,30,30)))+1.0
    
    for k in range(n):
        figExIm = plt.figure(figsize=(6,6))
        axExIm1 = figExIm.add_subplot(1,1,1)
        image1 = np.sum(image[k], axis=0)#+1.0
        masked_array1 = np.ma.array(image1, mask=(image1==0.0))
        im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                           norm=mpl.colors.LogNorm(), origin='lower')
        figExIm.patch.set_facecolor('white')
        axExIm1.title.set_text(model_title + ' {:d}'.format(k))
        axExIm1.set_xlabel('y [cells]', family='serif')
        axExIm1.set_ylabel('x [cells]', family='serif')
        figExIm.colorbar(im1)
        plt.savefig('./' + save_title+"_CollapseZ_{:d}.png".format(k))

        figExIm = plt.figure(figsize=(6,6))
        axExIm2 = figExIm.add_subplot(1,1,1)    
        image2 = np.sum(image[k], axis=1)#+1.0
        masked_array2 = np.ma.array(image2, mask=(image2==0.0))
        im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                           norm=mpl.colors.LogNorm(), origin='lower') 
        figExIm.patch.set_facecolor('white')
        axExIm2.title.set_text(model_title + ' {:d}'.format(k))
        axExIm2.set_xlabel('y [cells]', family='serif')
        axExIm2.set_ylabel('z [layers]', family='serif')
        figExIm.colorbar(im2)
        plt.savefig('./' + save_title+"_CollapseX_{:d}.png".format(k))

        figExIm = plt.figure(figsize=(6,6))
        axExIm3 = figExIm.add_subplot(1,1,1)
        image3 = np.sum(image[k], axis=2)#+1.0
        masked_array3 = np.ma.array(image3, mask=(image3==0.0))
        im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                           norm=mpl.colors.LogNorm(), origin='lower')
        figExIm.patch.set_facecolor('white')
        axExIm3.title.set_text(model_title + ' {:d}'.format(k))
        axExIm3.set_xlabel('x [cells]', family='serif')
        figExIm.colorbar(im3)
        plt.savefig('./' + save_title+"_CollapseY_{:d}.png".format(k))

    #print(np.min(image))
    

    figExIm = plt.figure(figsize=(6,6))
    axExIm1 = figExIm.add_subplot(1,1,1)
    image1 = np.mean(np.sum(image, axis=1), axis=0)#+1.0
    masked_array1 = np.ma.array(image1, mask=(image1==0.0))
    im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                       norm=mpl.colors.LogNorm(), origin='lower')
    figExIm.patch.set_facecolor('white')
    axExIm1.title.set_text(model_title + 'overlay')
    axExIm1.set_xlabel('y [cells]', family='serif')
    axExIm1.set_ylabel('x [cells]', family='serif')
    figExIm.colorbar(im1)
    plt.savefig('./' + save_title+"_CollapseZSum.png")

    figExIm = plt.figure(figsize=(6,6))
    axExIm2 = figExIm.add_subplot(1,1,1)    
    image2 = np.mean(np.sum(image, axis=2), axis=0)#+1.0
    masked_array2 = np.ma.array(image2, mask=(image2==0.0))
    im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                       norm=mpl.colors.LogNorm(), origin='lower') 
    figExIm.patch.set_facecolor('white')
    axExIm2.title.set_text(model_title + 'overlay')
    axExIm2.set_xlabel('y [cells]', family='serif')
    axExIm2.set_ylabel('z [layers]', family='serif')
    figExIm.colorbar(im2)
    plt.savefig('./' + save_title+"__CollapseXSum.png")
   
    figExIm = plt.figure(figsize=(6,6))
    axExIm3 = figExIm.add_subplot(1,1,1)    
    image3 = np.mean(np.sum(image, axis=3), axis=0)#+1.0
    masked_array3 = np.ma.array(image3, mask=(image3==0.0))
    im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                       norm=mpl.colors.LogNorm(), origin='lower')
    figExIm.patch.set_facecolor('white')
    axExIm3.title.set_text(model_title + 'overlay')
    axExIm3.set_xlabel('x [cells]', family='serif')
    axExIm3.set_ylabel('z [layers]', family='serif')
    figExIm.colorbar(im3)
    plt.savefig('./' + save_title+"_CollapseYSum.png")

    
    
    
def plt_NPFOs(data_real, data_fake, data_fake2, energy_center, maxN, minN, bins, xtitle,save_title):
    figSE = plt.figure(figsize=(8,8))
    axSE = figSE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
    
   
    pSEa = axSE.hist(data_real, bins=bins, range=[minN, maxN], density=None, edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')

    
    pSpnEb = axSE.hist(data_fake, bins=pSEa[1], range=None, density=None, edgecolor=color_list[1],
                   label = "orig", linewidth=linewidth_list[1], linestyle=linestyle_list[1],
                   histtype='step')
    
    pSpnEc = axSE.hist(data_fake2, bins=pSEa[1], range=None, density=None, edgecolor=color_list[2],
                   label = "orig", linewidth=linewidth_list[2], linestyle=linestyle_list[2],
                   histtype='step')
    
    axSE.set_xlabel(xtitle, family='serif')
    axSE.set_xlim([minN, maxN])
    axSE.set_ylim([0, 1800])
    posX= 0.02
    
    axSE.text(posX + 0.6, 0.65, "Geant4", horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes, color = color_list[0])
    
    axSE.text(posX + 0.6, 0.60, "WGAN", horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes, color = color_list[1])
    
    axSE.text(posX + 0.6, 0.55, "BiB-AE", horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes, color = color_list[2])
    
    axSE.text(posX,
            0.97,
            '{:d} GeV Pions'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes)

   
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
   
    figSE.patch.set_facecolor('white')

    #hep.cms.label(loc=0)
    plt.savefig(save_title+ str(energy_center)+"_single_NPFOs_comp.png")


def plt_nhits(data_real, data_fake, data_fake2, energy_center, maxN, minN, bins, ymax, ymin, xtitle, save_title):
    figSE = plt.figure(figsize=(8,8))
    axSE = figSE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
    
   
    pSEa = axSE.hist(data_real, bins=bins, range=[minN, maxN], density=None, edgecolor=color_list[0], 
                   weights=np.ones_like(data_real)/(float(len(data_real))),
                   label = "orig", linewidth=linewidth_list[0],color='lightgrey',
                   histtype='stepfilled')

    
    pSpnEb = axSE.hist(data_fake, bins=pSEa[1], range=None, density=None, edgecolor=color_list[1],
                   weights=np.ones_like(data_fake)/(float(len(data_fake))),
                   label = "orig", linewidth=linewidth_list[1], linestyle=linestyle_list[1],
                   histtype='step')
    
    pSpnEc = axSE.hist(data_fake2, bins=pSEa[1], range=None, density=None, edgecolor=color_list[2],
                   weights=np.ones_like(data_fake2)/(float(len(data_fake2))),
                   label = "orig", linewidth=linewidth_list[2], linestyle=linestyle_list[2],
                   histtype='step')
    
    axSE.set_xlabel(xtitle, family='serif')
    axSE.set_xlim([minN, maxN])
    axSE.set_ylim([ymin, ymax])
    posX= 0.03
    
    axSE.text(posX, 0.55, "Geant4", horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes, color = color_list[0])
    
    axSE.text(posX, 0.45, "WGAN", horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes, color = color_list[1])
    
    axSE.text(posX, 0.50, "BiB-AE", horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes, color = color_list[2])
    
    axSE.text(posX,
            0.97,
            '{:d} GeV Pions'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
             transform=axSE.transAxes)

   
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
   
    figSE.patch.set_facecolor('white')

    #hep.cms.label(loc=0)
    plt.savefig(save_title+ str(energy_center)+"_single_nhits_comp.png", dpi=300)



class CellIDDecoder:

    """ decoder for LCIO cellIDs """

    def __init__(self,encStr):
        self.encStr=encStr
        self.funs = {} 

        tokens = encStr.split(',')
        
        offset = 0
        
        for t in tokens:
        
         # print "token: " , t
        
          st = t.split(':')
        
          if len(st)==2:
            name = st[0]
            start = offset 
            width = int(st[1])
            offset += abs( width )
        
          elif len(st)==3:
            name = st[0]
            start = int(st[1]) 
            width = int(st[2])
            offset = start + abs( width )
        
        
          else:
            print ("unknown token:" , t)
        
          mask = int(0x0)
          for i in range(0,abs(width)):
            mask = mask | ( 0x1 << ( i + start) )
        
          setattr( CellIDDecoder , name , self.makefun( mask, start , width) )


    def makefun(self, mask,start,width):
      if( width > 0 ):
        return ( lambda ignore, cellID : (( mask & cellID) >> start )  )
      else:
        return ( lambda ignore, cellID : (~(( mask & cellID) >> start )  ^ 0xffffffff) )
