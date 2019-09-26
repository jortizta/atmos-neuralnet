
################
#
# COURTESY: Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Dataset handling
#
################


from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random
import pandas as pd

# global switch, use fixed max values for dim-less airfoil data?
fixedAirfoilNormalization = True
# global switch, make data dimensionless?
makeDimLess = True
# global switch, remove constant offsets from pressure channel?
removePOffset = True

## helper - compute absolute of inputs or targets
def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval


class TurbDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, dataProp=None, mode=TRAIN, dataDir="../data/sliceMean/", dataDirTest="../data/sliceMean/", shuffle=0, normMode=0, nSample=5):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """        

        self.inputs=np.empty((int((nSample-1)*256),7,128,1))
        self.targets=np.empty((int((nSample-1)*256),2,128,1))
        rawData = []
        for t in range(1,nSample):
          for snum in range(256):
            # changed from csv to npy files
            f = np.load(dataDir+'train_t_'+str(t).zfill(2)+'_slice_'+str(snum).zfill(3)+'.npy')# pd.read_csv(dataDir+'train_t_'+str(t).zfill(2)+'_slice_'+str(snum).zfill(3)+'.csv', header=None)
            rawData.append(f)

        rawData = np.array(rawData)
        np.random.shuffle(rawData) # shuffle to prevent coherent data input

        rawDataMean=np.mean(rawData,axis=0)
        rawDataF= rawData-rawDataMean[np.newaxis,:,:]
        sDev=np.std(rawData,axis=0)
        maxRaw = np.max(rawData, axis=0)
        minRaw = np.min(rawData, axis=0)

        # rawDataNorm= rawDataF/sDev[np.newaxis,:,:]
        rawDataNorm = (rawDataF-minRaw[np.newaxis,:,:]) / (maxRaw[np.newaxis,:,:]-minRaw[np.newaxis,:,:])


        rawDataNorm[np.isnan(rawDataNorm)] = 0
        rawDataNorm[np.isinf(rawDataNorm)] = 0

        self.inputs[:,0:7,:,0]=rawDataNorm[:,0:7,0:128]
        self.targets[:,:,:,0]=rawDataNorm[:,7:,0:128]
        self.totalLength = self.inputs.shape[0]



        if not (mode==self.TRAIN or mode==self.TEST):
            print("Error - TurbDataset invalid mode "+format(mode) ); exit(1)

        if normMode==1: 
            print("Warning - poff off!!")
            removePOffset = False
        if normMode==2: 
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest # only for mode==self.TEST


        if not self.mode==self.TEST:
            # split for train/validation sets (80/20) , max 400

            targetLength = self.totalLength - min( int(self.totalLength*0.2) , 400)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = targetLength


    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    #  reverts normalization 
    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0,:,:] /= (1.0/self.max_targets_0)
        a[1,:,:] /= (1.0/self.max_targets_1)
        a[2,:,:] /= (1.0/self.max_targets_2)

        if makeDimLess:
            a[0,:,:] *= v_norm**2
            a[1,:,:] *= v_norm
            a[2,:,:] *= v_norm
        return a

# simplified validation data set (main one is TurbDataset above)


class ValiDataset(TurbDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

