#-*- coding: UTF-8 -*- 
import os
import shutil
import numpy as np
import scipy.io as scio
import cv2
import argparse

# Process an image from DOTA dataset
def processDOTA(imgReadName, labelReadName, imgWritePath, labelWritePath):
    # define variables
    gsd = 0 
    numCar = 0 
    numSmallCar = 0 
    downSampleTimes = 0 

    # anchor coord list
    x1=[]; x2=[]; x3=[]; x4=[]
    y1=[]; y2=[]; y3=[]; y4=[]
    # point coord list
    xLabel=[]; yLabel=[]

    # read file
    with open(labelReadName,'r') as labelRead:
        temp = labelRead.readline() # first line
        temp = labelRead.readline() # second line: GSD
        temp = temp.strip().split(':') 
        if temp[1] == 'null': # discard the image without GSD data
            return
        gsd = float(temp[1]) 
        if gsd > 0.89: # discard the image without a GSD larger than 0.89
            return
        
        while True:
            lines = labelRead.readline() 
            temp = lines.strip().split() 
            if not lines: #
                break
            if temp[8] == 'small-vehicle':
                numSmallCar += 1
            if temp[8] == 'small-vehicle' or temp[8] == 'large-vehicle': 
                x1.append(float(temp[0])); y1.append(float(temp[1]))
                x2.append(float(temp[2])); y2.append(float(temp[3]))
                x3.append(float(temp[4])); y3.append(float(temp[5]))
                x4.append(float(temp[6])); y4.append(float(temp[7]))
                numCar += 1

    # discard the image without small vehicles
    if numSmallCar == 0:
        return

    img=cv2.imread(imgReadName)
    # downsampling the image with a GSD smaller than 0.5
    if gsd <= 0.5: 
        while True:
            img = cv2.pyrDown(img)
            gsd *= 2 
            downSampleTimes += 1 
            if gsd >= 0.55: 
                break

    # discard the image with too-small size after downsampling
    size = img.shape
    H = size[0]; W = size[1]
    if H <= 200 or W <= 200:
        return

    # get the center points of anchors
    for i in range(0,numCar):
        
        x_mark = (x1[i]+x2[i]+x3[i]+x4[i])/4
        y_mark = (y1[i]+y2[i]+y3[i]+y4[i])/4
        
        x_mark /= 2**downSampleTimes
        y_mark /= 2**downSampleTimes
       
        xLabel.append(x_mark); yLabel.append(y_mark)

    # split the image into slices with a size of 200*200 ~ 1024*1024
    yNum = H//1024; xNum = W//1024 
    cellImage = []; cellxLabel=[]; cellyLabel=[] 
    if yNum > 0 or xNum > 0: 
        for i in range(0, xNum):
            for j in range(0, yNum):
                tmpImage = img[j*1024:(j+1)*1024, i*1024:(i+1)*1024] 
                tmpxLabel=[]; tmpyLabel=[] 
                for k in range(0, numCar):
                    tmpx=xLabel[k]; tmpy=yLabel[k]
                    if tmpx>i*1024 and tmpx<=(i+1)*1024 and tmpy>j*1024 and tmpy<=(j+1)*1024: 
                        tmpxLabel.append(tmpx-i*1024); tmpyLabel.append(tmpy-j*1024)
                cellImage.append(tmpImage); cellxLabel.append(tmpxLabel); cellyLabel.append(tmpyLabel)

            if H % 1024 > 200:
                tmpImage = img[yNum*1024:H, i*1024:(i+1)*1024]
                tmpxLabel=[]; tmpyLabel=[]
                for k in range(0, numCar):
                    tmpx=xLabel[k]; tmpy=yLabel[k]
                    if tmpx>i*1024 and tmpx<=(i+1)*1024 and tmpy>yNum*1024 and tmpy<=H:
                        tmpxLabel.append(tmpx-i*1024); tmpyLabel.append(tmpy-yNum*1024)
                cellImage.append(tmpImage); cellxLabel.append(tmpxLabel); cellyLabel.append(tmpyLabel)
                
        if W % 1024 > 200:
            for j in range(0, yNum):
                tmpImage = img[j*1024:(j+1)*1024, xNum*1024:W]
                tmpxLabel=[]; tmpyLabel=[]
                for k in range(0, numCar):
                    tmpx=xLabel[k]; tmpy=yLabel[k]
                    if tmpx>xNum*1024 and tmpx<=W and tmpy>j*1024 and tmpy<=(j+1)*1024:
                        tmpxLabel.append(tmpx-xNum*1024); tmpyLabel.append(tmpy-j*1024)
                cellImage.append(tmpImage); cellxLabel.append(tmpxLabel); cellyLabel.append(tmpyLabel)

            if H % 1024 > 200:
                tmpImage = img[yNum*1024:H, xNum*1024:W]
                tmpxLabel=[]; tmpyLabel=[]
                for k in range(0, numCar):
                    tmpx=xLabel[k]; tmpy=yLabel[k]
                    if tmpx>xNum*1024 and tmpx<=W and tmpy>yNum*1024 and tmpy<=H:
                        tmpxLabel.append(tmpx-xNum*1024); tmpyLabel.append(tmpy-yNum*1024)
                cellImage.append(tmpImage); cellxLabel.append(tmpxLabel); cellyLabel.append(tmpyLabel)
    else:
        cellImage.append(img); cellxLabel.append(xLabel); cellyLabel.append(yLabel)

    # output to files
    imgName = imgReadName.split('.')[-2].split('/')[-1].split('\\')[-1]
    labelName = labelReadName.split('.')[-2].split('/')[-1].split('\\')[-1]

    for i in range(0,len(cellImage)):
        numCar_cell = len(cellxLabel[i])
       
        if numCar_cell == 0:
            continue

        imgWriteName = os.path.join(imgWritePath, imgName + '_' + str(i) + '.jpg')
        labelWriteName = os.path.join(labelWritePath, labelName + '_' + str(i) + '.txt')

        if yNum == 0 and xNum == 0:
            imgWriteName = os.path.join(imgWritePath, imgName + '.jpg')
            labelWriteName = os.path.join(labelWritePath, labelName + '.txt')

        # output the point labels into a txt file
        """
        Text Format：
        Line 1：GSD after downsampling
        Line 2：Number of cars
        Line 3 and after：Coordinates of center points of cars

        e.g.
        GSD:1.14514
        numCar:1919
        3216.7 1049.2
        ...
        """

        with open(labelWriteName,'w') as labelWrite:
            labelWrite.write("GSD:" + str(gsd) + "\n")
            labelWrite.write("numCar:" + str(numCar_cell) + "\n")
            
            for j in range(0, numCar_cell):
                labelWrite.write(str(cellxLabel[i][j]) + " " + str(cellyLabel[i][j]))
                if j < numCar_cell-1:
                    labelWrite.write("\n")
        
        # output image
        cv2.imwrite(imgWriteName, cellImage[i])

# Process an image from ITCVD dataset
def processITCVD(imgReadName, labelReadName, imgWriteName, labelWriteName):
    gsd = 0.1 
    numCar = 0 
    downSampleTimes = 3 
    
    xLabel = []; yLabel = []

    fileName = imgReadName.split('.')[-2].split('/')[-1].split('\\')[-1]

    if(int(fileName) >= 71): #00071 and after have oblique view and should be discarded
        return

    # read .mat file
    data = scio.loadmat(labelReadName)
    data = data['x'+fileName]
    numCar = np.shape(data)[0]

    # downsampling
    img=cv2.imread(imgReadName)

    for k in range(0,3):
        img=cv2.pyrDown(img)
        gsd*=2

    # get the center points of anchors
    for i in range(0,numCar):
        x_mark=(data[i][0]+data[i][2])/2
        y_mark=(data[i][1]+data[i][3])/2

        x_mark/=2**downSampleTimes
        y_mark/=2**downSampleTimes
        
        xLabel.append(x_mark); yLabel.append(y_mark)

    # output the point labels into a txt file
    """
    Text Format：
    Line 1：GSD after downsampling
    Line 2：Number of cars
    Line 3 and after：Coordinates of center points of cars

    e.g.
    GSD:1.14514
    numCar:1919
    3216.7 1049.2
    ...
    """
    with open(labelWriteName,'w') as labelWrite:
        labelWrite.write("GSD:"+str(gsd)+"\n")
        labelWrite.write("numCar:"+str(numCar)+"\n")
        for j in range(0,numCar):
            labelWrite.write(str(xLabel[j])+" "+str(yLabel[j]))
            if j<numCar-1:
                labelWrite.write("\n")

    # output image
    cv2.imwrite(imgWriteName,img)


# Get the file roots of input and output datasets
parser = argparse.ArgumentParser(description='Process DOTA and ITCVD datasets, turning the anchor-box-annotations into point-annotations to get RSVC2021 dataset.')
parser.add_argument('--DOTA_ROOT', default='./DOTA/', type=str, help='File root of original DOTA dataset')
parser.add_argument('--ITCVD_ROOT', default='./ITCVD/', type=str, help='File root of original ITCVD dataset')
parser.add_argument('--OUTPUT_ROOT', default='./Merge/', type=str, help='Output root of RSVC2021 dataset')

args = parser.parse_args()
DOTA_ROOT = args.DOTA_ROOT
ITCVD_ROOT = args.ITCVD_ROOT
OUTPUT_ROOT = args.OUTPUT_ROOT

DOTA_TRAIN_IMG = os.path.join(DOTA_ROOT, 'train/images')
DOTA_TRAIN_LABEL = os.path.join(DOTA_ROOT, 'train/labelTxt')
DOTA_VAL_IMG = os.path.join(DOTA_ROOT, 'val/images')
DOTA_VAL_LABEL = os.path.join(DOTA_ROOT, 'val/labelTxt')

ITCVD_TRAIN_IMG = os.path.join(ITCVD_ROOT, 'Training/Image')
ITCVD_TRAIN_LABEL = os.path.join(ITCVD_ROOT, 'Training/GT')
ITCVD_TEST_IMG = os.path.join(ITCVD_ROOT, 'Testing/Image')
ITCVD_TEST_LABEL = os.path.join(ITCVD_ROOT, 'Testing/GT')

# Deploy the output structure
RSVC_TRAIN = os.path.join(OUTPUT_ROOT, 'train')
RSVC_TEST = os.path.join(OUTPUT_ROOT,'test')
RSVC_TRAIN_IMG = os.path.join(RSVC_TRAIN, 'img')
RSVC_TRAIN_LABEL = os.path.join(RSVC_TRAIN, 'label')
RSVC_TRAIN_DEN = os.path.join(RSVC_TRAIN, 'den') # To generate density map, please refer to the C-3-Framework: https://github.com/gjy3035/C-3-Framework
RSVC_TEST_IMG = os.path.join(RSVC_TEST, 'img')
RSVC_TEST_LABEL = os.path.join(RSVC_TEST, 'label')
RSVC_TEST_DEN = os.path.join(RSVC_TEST, 'den')

if not os.path.exists(RSVC_TRAIN):
    os.makedirs(RSVC_TRAIN)
if not os.path.exists(RSVC_TEST):
    os.makedirs(RSVC_TEST)
if not os.path.exists(RSVC_TRAIN_IMG):
    os.makedirs(RSVC_TRAIN_IMG)
if not os.path.exists(RSVC_TRAIN_LABEL):
    os.makedirs(RSVC_TRAIN_LABEL)
if not os.path.exists(RSVC_TRAIN_DEN):
    os.makedirs(RSVC_TRAIN_DEN)
if not os.path.exists(RSVC_TEST_IMG):
    os.makedirs(RSVC_TEST_IMG)
if not os.path.exists(RSVC_TEST_LABEL):
    os.makedirs(RSVC_TEST_LABEL)
if not os.path.exists(RSVC_TEST_DEN):
    os.makedirs(RSVC_TEST_DEN)


# Get the file lists of each dataset
DOTA_TRAIN_FILE_LIST = [filename for root, dirs, filename in os.walk(DOTA_TRAIN_IMG)]
DOTA_VAL_FILE_LIST = [filename for root, dirs, filename in os.walk(DOTA_VAL_IMG)]
ITCVD_TRAIN_FILE_LIST = [filename for root, dirs, filename in os.walk(ITCVD_TRAIN_IMG)]
ITCVD_TEST_FILE_LIST = [filename for root, dirs, filename in os.walk(ITCVD_TEST_IMG)]

# Process the DOTA training set
numProc = 1; numAll = len(DOTA_TRAIN_FILE_LIST[0])
for fname in DOTA_TRAIN_FILE_LIST[0]:
    filename_no_ext = fname.split('.')[0]
    imgReadName = os.path.join(DOTA_TRAIN_IMG, filename_no_ext + ".png")
    labelReadName = os.path.join(DOTA_TRAIN_LABEL, filename_no_ext + ".txt")
    imgWritePath = RSVC_TRAIN_IMG
    labelWritePath = RSVC_TRAIN_LABEL
    processDOTA(imgReadName, labelReadName, imgWritePath, labelWritePath)
    if numProc%10==0:
        print("processing DOTA TRAIN dataset: "+str(numProc)+"/"+str(numAll))
    if numProc==numAll:
        print("processing DOTA TRAIN dataset: "+str(numProc)+"/"+str(numAll))
        print("DOTA TRAIN dataset processing complete!")
    numProc+=1

# Process the DOTA testing set
numProc = 1; numAll = len(DOTA_VAL_FILE_LIST[0])
for fname in DOTA_VAL_FILE_LIST[0]:
    filename_no_ext = fname.split('.')[0]
    imgReadName = os.path.join(DOTA_VAL_IMG, filename_no_ext + ".png")
    labelReadName = os.path.join(DOTA_VAL_LABEL, filename_no_ext + ".txt")
    imgWritePath = RSVC_TRAIN_IMG
    labelWritePath = RSVC_TRAIN_LABEL
    processDOTA(imgReadName, labelReadName, imgWritePath, labelWritePath)
    if numProc%10==0:
        print("processing DOTA TEST dataset: "+str(numProc)+"/"+str(numAll))
    if numProc==numAll:
        print("processing DOTA TEST dataset: "+str(numProc)+"/"+str(numAll))
        print("DOTA TEST dataset processing complete!")
    numProc+=1

# Process the ITCVD training set
numProc = 1; numAll = len(ITCVD_TRAIN_FILE_LIST[0])
for fname in ITCVD_TRAIN_FILE_LIST[0]:
    filename_no_ext = fname.split('.')[0]
    imgReadName = os.path.join(ITCVD_TRAIN_IMG, filename_no_ext + ".jpg")
    labelReadName = os.path.join(ITCVD_TRAIN_LABEL, filename_no_ext + ".mat")
    imgWriteName = os.path.join(RSVC_TRAIN_IMG, filename_no_ext + ".jpg")
    labelWriteName = os.path.join(RSVC_TRAIN_LABEL, filename_no_ext + ".txt")
    processITCVD(imgReadName, labelReadName, imgWriteName, labelWriteName)
    if numProc%10 == 0:
        print("processing ITCVD TRAIN dataset: "+str(numProc)+"/"+str(numAll))
    if numProc == numAll:
        print("processing ITCVD TRAIN dataset: "+str(numProc)+"/"+str(numAll))
        print("ITCVD TRAIN dataset processing complete!")
    numProc += 1

# Process the ITCVD testing set
numProc = 1; numAll = len(ITCVD_TEST_FILE_LIST[0])
for fname in ITCVD_TEST_FILE_LIST[0]:
    filename_no_ext = fname.split('.')[0]
    imgReadName = os.path.join(ITCVD_TEST_IMG, filename_no_ext + ".jpg")
    labelReadName = os.path.join(ITCVD_TEST_LABEL, filename_no_ext + ".mat")
    imgWriteName = os.path.join(RSVC_TRAIN_IMG, filename_no_ext + ".jpg")
    labelWriteName = os.path.join(RSVC_TRAIN_LABEL, filename_no_ext + ".txt")
    processITCVD(imgReadName, labelReadName, imgWriteName, labelWriteName)
    if numProc%10 == 0:
        print("processing ITCVD TEST dataset: "+str(numProc)+"/"+str(numAll))
    if numProc == numAll:
        print("processing ITCVD TEST dataset: "+str(numProc)+"/"+str(numAll))
        print("ITCVD TEST dataset processing complete!")
    numProc += 1

# Read the lists of training and testing set
train_List = []
test_List = []
with open('train.list','r') as trainRead:
    train_List = trainRead.read().splitlines()
with open('test.list','r') as testRead:
    test_List = testRead.read().splitlines()

# Sorting images and labels according to lists
print("Sorting images and labels...")
RSVC_FILE_LIST = [filename for root, dirs, filename in os.walk(RSVC_TRAIN_IMG)]
for fname in RSVC_FILE_LIST[0]:
    filename_no_ext = fname.split('.')[0]
    procFlag = 0
## If an image is in the testing set, move the image and label files to the corresponding directories
    for fnTest in test_List:
        if filename_no_ext == fnTest:
            shutil.move(os.path.join(RSVC_TRAIN_IMG, filename_no_ext + '.jpg'), RSVC_TEST_IMG)
            shutil.move(os.path.join(RSVC_TRAIN_LABEL, filename_no_ext + '.txt'), RSVC_TEST_LABEL)
            procFlag = 1
            break
## Else if an image is in the training set, keep the image and label files
    if procFlag == 0:
        for fnTrain in train_List:
            if filename_no_ext == fnTrain:
                procFlag = 1
                break
## Else, i.e., an image is not in the training or testing set, remove the image and label files
    if procFlag == 0:
        os.remove(os.path.join(RSVC_TRAIN_IMG, filename_no_ext + '.jpg'))
        os.remove(os.path.join(RSVC_TRAIN_LABEL, filename_no_ext + '.txt'))

print("RSVC2021 dataset generating complete!")