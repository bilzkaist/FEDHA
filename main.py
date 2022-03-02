#*****************************************************************************
#
#                            TEMPLATE FEDHA Code.
#                             Written  by Bilal Dastagir.
#                                March , 1st, 2022
#
#******************************************************************************

from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from time import time
from skimage.feature import plot_matches
from skimage.transform import pyramid_gaussian

from utils import *

import time as tm

#%matplotlib inline


#Global Variables 
BETA = [0]
ALPHA = [1]
BRAVO  = [2]
CHARLIE = [3]

DIRPATH = "/Users/bilaldastagir/Documents/vscode/FEDHA/"

# multi-scale
N_LAYERS = 1
DOWNSCALE = 2

def run_fedha():
    print("FEDHA Program is Started........... !!!")
    # Write code Here
    start = tm.time()
    original_img1 = img1 = Image.open(DIRPATH + 'images/KAIST.jpeg')
    gray1 = img1.convert('L')
    gray1 = gray1.filter(ImageFilter.GaussianBlur(radius = 3))
    gray1 = np.array(gray1)
    grays1 = list(pyramid_gaussian(gray1, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))

    original_img2 = img2 = Image.open(DIRPATH + 'images/KAIST2.jpeg')
    gray2 = img2.convert('L')
    gray2 = np.array(gray2)
    grays2 = list(pyramid_gaussian(gray2, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))

    np.set_printoptions(threshold=np.inf)
    print("Gray1 with lenght (",len(gray1),") : ",gray2)
    print("Gray2 with lenght (",len(gray2),") : ",gray2)
    lap1 = tm.time()
    executionTime = (lap1 - start)
    print("Images are read - The Execution Time = ", executionTime, " Seconds")
    
    scales = [DOWNSCALE**i for i in range(N_LAYERS)]
    print('scales: ', scales, '\n')
    features_img1 = Image.fromarray(np.copy(img1))
    features_img2 = Image.fromarray(np.copy(img2))
    print("Feature Image : ", features_img1)

    kps1 = [] 
    kps2 = []

    ds1 = []
    ds2 = []
    lap2 = tm.time()
    executionTime = (lap2 - lap1)
    print("Initialization Variables Completed  - The Execution Time = ", executionTime, " Seconds")

    for i in range(N_LAYERS):
        lapLoopStart = tm.time()
        print('pyramid layer: ', i)
        print('scales[i]: ', scales[i])
        scale_coeff1 = (gray1.shape[1]/grays1[i].shape[1], gray1.shape[0]/grays1[i].shape[0])
        scale_coeff2 = (gray2.shape[1]/grays2[i].shape[1], gray2.shape[0]/grays2[i].shape[0])
        lapLoop2 = tm.time()
        executionTime = (lapLoop2 - lapLoopStart)
        print("Images Scaling Coeffecients Completed  - The Execution Time = ", executionTime, " Seconds")

        print('scale_coeff1: ', scale_coeff1)
        print('scale_coeff2: ', scale_coeff2)
        
        print('grays1[i] shape: ', grays1[i].shape)
        print('grays2[i] shape: ', grays2[i].shape)
        
        scale_kp1 = FAST(grays1[i], N=9, threshold=0.1, nms_window=3)
        scale_kp2 = FAST(grays2[i], N=9, threshold=0.1, nms_window=3)
        lapLoop3 = tm.time()
        executionTime = (lapLoop3 - lapLoop2)
        print("Images Scaling FAST Processing Completed  - The Execution Time = ", executionTime, " Seconds")

        print('kp1: ', len(scale_kp1))
        print('kp2: ', len(scale_kp2))
        
        for keypoint in scale_kp1:
    #         features_img1 = cv2.circle(features_img1, tuple(np.round(keypoint*scale_coeff1).astype(np.int32)), int(3*scales[i]), (0,255,0), 1)
            x0 = np.round(keypoint*scale_coeff1)[0]-3*scales[i]
            y0 = np.round(keypoint*scale_coeff1)[1]-3*scales[i]
            x1 = np.round(keypoint*scale_coeff1)[0]+3*scales[i]
            y1 = np.round(keypoint*scale_coeff1)[1]+3*scales[i]
            draw1 = ImageDraw.Draw(features_img1)
            draw1.ellipse([x0,y0,x1,y1], outline='red', width=1)
        lapLoop4 = tm.time()
        executionTime = (lapLoop4 - lapLoop3)
        print("Image 1 Extraction Completed  - The Execution Time = ", executionTime, " Seconds")

        for keypoint in scale_kp2:
    #         features_img2 = cv2.circle(features_img2, tuple(np.round(keypoint*scale_coeff2).astype(np.int32)), int(3*scales[i]), (0,255,0), 1)
            x0 = np.round(keypoint*scale_coeff2)[0]-3*scales[i]
            y0 = np.round(keypoint*scale_coeff2)[1]-3*scales[i]
            x1 = np.round(keypoint*scale_coeff2)[0]+3*scales[i]
            y1 = np.round(keypoint*scale_coeff2)[1]+3*scales[i]
            draw2 = ImageDraw.Draw(features_img2)
            draw2.ellipse([x0,y0,x1,y1], outline='red', width=1)
        lapLoop5 = tm.time()
        executionTime = (lapLoop5 - lapLoop4)
        print("Image 2 Extraction Completed  - The Execution Time = ", executionTime, " Seconds")
    
        lapLoopPlotStart = tm.time()
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.imshow(grays1[i], cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(features_img1)

        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.imshow(grays2[i], cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(features_img2)
        lapLoopPlotEnd = tm.time()
        
    #     orientations1, orientations2 = None, None
        orientations1 = corner_orientations(grays1[i], scale_kp1)
        orientations2 = corner_orientations(grays2[i], scale_kp2)
        lapLoop6 = tm.time()
        executionTime = (lapLoop6 - lapLoopPlotEnd)
        print("Images Orientation Processing Completed  - The Execution Time = ", executionTime, " Seconds")

        
        if not (len(scale_kp1) == 0 or len(scale_kp2) == 1):
            # avoding errors when no keypoints are computed (happens with small scales - higher pyramid layers)
            kps1.append(np.round(scale_kp1*scale_coeff1).astype(np.int32))
            kps2.append(np.round(scale_kp2*scale_coeff2).astype(np.int32))
            
            d1 = BRIEF(grays1[i], scale_kp1, orientations=orientations1, mode='uniform', n=128)
            d2 = BRIEF(grays2[i], scale_kp2, orientations=orientations2, mode='uniform', n=128)
            ds1.append(d1)
            ds2.append(d2)
            lapLoop7 = tm.time()
            executionTime = (lapLoop7 - lapLoop6)
            print("Images Error Avoiding Completed  - The Execution Time = ", executionTime, " Seconds")

           
        plt.show()
        print('-'*50)
 
    stop = tm.time()
    executionTimeFinal = (stop - start)
    print("The Feature Extraction Final Execution Time = ", executionTimeFinal, " Seconds")  
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(features_img1)
    plt.subplot(1,2,2)
    plt.imshow(features_img2)
    plt.show()
    print("FEDHA Program is Ended Successfully !!!")


def run_beta():
    print("Beta Program is Started........... !!!")
    # Write code Here
    run_fedha()
    print("Beta Program is Ended Successfully !!!")
    return BETA

def run_alpha():
    print("Alpha Program is Started........... !!!")
    # Write code Here
    print("Alpha Program is Ended Successfully !!!")
    return ALPHA
    

def run_bravo():
    print("Bravo Program is Started........... !!!")
    # Write code Here
    print("Bravo Program is Ended Successfully !!!")
    return BRAVO

def run_charlie():
    print("Charlie Program is Started........... !!!")
    # Write code Here
    print("Charlie Program is Ended Successfully !!!")
    return CHARLIE
    
def switch_mode(mode):
    # Program Started
    switcher = {
        0: run_beta,
        1: run_alpha,
        2: run_bravo,
        3: run_charlie
        
        
    }
     # Get the function from switcher dictionary
    func = switcher.get(mode, lambda: "Invalid mode")
    # Execute the function
    print("Mode Selected : ",func())
    # Program Ended 

def run():
    print("......................Main Program is Started........... !!!\n")
    # write coode here
    runMode = BETA
    if (runMode == ALPHA):
        run_alpha()
    else:
        run_beta()
        
    print("\n......................Main Program is Ended Successfully !!!")

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def print_bye(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Bye, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi(' Bilal Dastagir')
    run()
    print_bye('Bilal Dastagir')