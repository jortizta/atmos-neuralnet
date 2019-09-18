
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

    def __init__(self, dataProp=None, mode=TRAIN, dataDir="../data/", dataDirTest="../data/", shuffle=0, normMode=0, nSample=5):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """        

        self.inputs=np.empty((int(nSample*256),7,128,1))
        self.targets=np.empty((int(nSample*256),2,128,1))
        zm = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430,
                440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
                580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710,
                720, 730, 740, 750, 760, 770, 780, 789.0909, 797.3553, 804.8685,
                811.6987, 817.9079, 823.5526, 828.6842, 833.6842, 838.6842, 843.6842,
                848.6842, 853.6842, 858.6842, 863.6842, 868.6842, 873.6842, 878.6842,
                883.6842, 888.6842, 893.6842, 898.6842, 903.6842, 908.6842, 913.6842,
                918.6842, 923.6842, 928.6842, 934.1842, 940.2342, 946.8892, 954.2097,
                962.2623, 971.1201, 980.8636, 991.5815, 1003.371, 1016.34, 1030.606,
                1046.298, 1063.559, 1082.547, 1103.433, 1126.408, 1151.68, 1179.48,
                1210.059, 1243.697, 1280.698, 1321.399, 1366.171, 1415.419, 1469.593])
        rawData = []
        for t in range(nSample):
          for snum in range(256):
            # changed from csv to npy files
            f = np.load(dataDir+'train_t_'+str(t).zfill(2)+'_slice_'+str(snum).zfill(3)+'.npy')# pd.read_csv(dataDir+'train_t_'+str(t).zfill(2)+'_slice_'+str(snum).zfill(3)+'.csv', header=None)
            rawData.append(f)

        rawData = np.array(rawData)
        thl = rawData[:,3,:]
        qt  = rawData[:,4,:]
        thl_gradient = np.gradient(thl,zm,axis=1)     
        qt_gradient = np.gradient(qt,zm,axis=1)
        rawData = np.concatenate((rawData,thl_gradient[:,np.newaxis,:]),axis=1)
        rawData = np.concatenate((rawData,qt_gradient[:,np.newaxis,:]),axis=1)
        np.random.shuffle(rawData) # shuffle to prevent coherent data input

        rawDataMean=np.mean(rawData,axis=0)
        rawDataF= rawData-rawDataMean[np.newaxis,:,:]
        sDev=np.std(rawData,axis=0)
        rawDataNorm= rawDataF/sDev[np.newaxis,:,:]

        rawDataNorm[np.isnan(rawDataNorm)] = 0

        self.inputs[:,0:5,:,0]=rawDataNorm[:,0:5,0:128]
        self.inputs[:,5:7,:,0]=rawDataNorm[:,7:9,0:128]
        self.targets[:,:,:,0]=rawDataNorm[:,5:7,0:128]
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

