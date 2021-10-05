'''Author: Yogesh Verma'''
'''Shower Resolution using Deep Learning'''

import matplotlib.pyplot as plt
import numpy as np
import pandas
import ROOT
from sklearn.preprocessing import MinMaxScaler
import random
def get_pi0():
    final_data = []
    data_filename = '/home/datab/Resolution_Data/B5_pi0_5GeV_100k.root'
    inFile = ROOT.TFile.Open(data_filename,"READ")
    tree = inFile.Get("B5")
    N=tree.GetEntries()
    for entrynum in range(0,N):
        hit_cells = []
        energy = []
        print("Events read: ",entrynum)
        tree.GetEntry(entrynum)
        edep_hit = getattr(tree,"ECEnergy")
        edep_hit_vec = getattr(tree,"ECEnergyVector")
        edep = []
        for i in range(18000):
            edep.append(edep_hit_vec[i])

        Energy_dep = np.array(edep)   
        Energy = np.reshape(Energy_dep,(30,30,20))
        final_data.append(Energy)

    labels = [0 for i in range(N)]
    return final_data,labels
