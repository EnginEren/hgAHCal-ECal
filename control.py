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

