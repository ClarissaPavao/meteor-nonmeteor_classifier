#
# camss file sorter version 1.0
#
# author: Clarissa Pavao
# last updated : 8/4/2022
#------------------------------------------------------------------

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import sys, string, os, errno
from subprocess import call
import json
import operator
import os.path
import csv
import math
from os import listdir
from os.path import isfile, join
import io
from shutil import copyfile
import smtplib
import shutil
from zipfile import ZipFile
#from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras import datasets, layers, models
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#INPUT -------------------------------------------------------------
#folder names to be processed (date or partial date)
folder_name = '2013_12'
#portable drive directory 
input_dir = 'E:\CAMSS\Data\\'

#computer directory
comp_dir = 'C:\CAMS\Data\\'
#folders to be created under date folder
sp_dir = 'SpectralFiles'
ff_dir = 'FF_Files'


#define parameter file
def read():
    params = []
    with open('C:\CAMS\Runfolder\spff_params.txt',  'r') as f:
        for line in f:
            params.append(line.split())

    paramdict = dict()
    for param in params:
        paramdict[param[0]] = param[2]

    return paramdict
print("Reading in parameters from the text file.")
params = read()


#move zip files from drive to computer
cont = os.listdir(input_dir)
for folders in cont:
    print(folders)
    if folders.startswith(folder_name):
        shutil.move(input_dir + folders, comp_dir + folders)

    
#unzip files on computer Data folder for each date to a new common date folder
comp_list_dir = os.listdir(comp_dir)        
for folder in comp_list_dir:
    if folder.startswith(folder_name):
        files_in_folder = os.listdir(comp_dir + folder)
        
        for zipped_file in files_in_folder:
            print(zipped_file)
            if zipped_file.endswith('.zip'):
                z = ZipFile(comp_dir + folder + '/' + zipped_file)
                z.extractall(comp_dir + folder)


#create SpectralFiles and FF_Files folders
        files1_in_folder = os.listdir(comp_dir + folder)        
        for new_folder in files1_in_folder:
            if new_folder.endswith('zip'):
                break
            else:
                new_folder_dir = os.listdir(comp_dir + folder + '/' + new_folder)
                sp_path = os.path.join(comp_dir + folder + '/' + new_folder + '/' , sp_dir)      # join the main file (date) to the SpectralFiles
                ff_path = os.path.join(comp_dir + folder + '/' + new_folder + '/' , ff_dir)      # join the main file (date) to the FF_Files
                os.makedirs(sp_path, exist_ok=True)                                              # make SpectralFiles folder
                os.makedirs(ff_path, exist_ok=True)                                              # make FF_Files folder
            print('Finished with making SpectralFiles folder and FF_Files folder')    
            
                
#move sp files to SpectralFiles and ff files to FF_Files
            for file in new_folder_dir:
                if file.startswith('SP') and file.endswith('.bin') and os.path.isfile(comp_dir + folder + '/' + new_folder + '/' + sp_dir + '/' + file) == False:
                    shutil.move(comp_dir + folder + '/' + new_folder +'/' + file, comp_dir + folder + '/' + new_folder + '/' + sp_dir )
                if file.startswith('FF') and file.endswith('.bin') and os.path.isfile(comp_dir + folder + '/' + new_folder + '/' + ff_dir + '/' + file) == False:
                    shutil.move(comp_dir + folder + '/' + new_folder +'/' + file, comp_dir + folder + '/' + new_folder + '/' + ff_dir )
            print('Done with moving SP files and FF files.')
                
            
#change directory to SpectralFiles to create jpgs
            sp_file_list = comp_dir + folder + '/' + new_folder + '/' + sp_dir 
            sp_file_dir = os.listdir(sp_file_list) 
#reads in the SP6... files and the directory they are located 
            file_name = 'SP6'
            for spec in sp_file_dir:
                if spec.startswith(file_name) and spec.endswith('.bin'):
                    currentDir = spec
#create a list of the files so single file viewer can iterate through each one at a time
                    extracted = []
                    extracted.append(currentDir)
#Loops through files and saves as jpgs
                    for directory in set(extracted):
                        call(['C:\CAMS\Runfolder\SPFF_SingleFileViewer', sp_file_list + '\\' + directory,
                            params['nfiles_write']])
            print('Done with converting .bins to jpgs.')
            
                    
#move all maxpixel to new folder called SpectralFilesJPGs and remove all the rest
            updated_sp_file_dir = os.listdir(comp_dir + folder + '/' + new_folder + '/' + sp_dir) 
            new_sp_dir = 'SpectralFilesJPGs'
            new_sp_path = os.path.join(comp_dir + folder + '/' + new_folder + '/', new_sp_dir)
            os.makedirs(new_sp_path, exist_ok = True)
            for new_spec in updated_sp_file_dir:
                if new_spec.endswith('_maxpixel.jpg'):
                    shutil.move(comp_dir + folder + '/' + new_folder + '/' + sp_dir + '/' + new_spec, comp_dir + folder + '/' + new_folder + '/' + new_sp_dir + '/' + new_spec)
            updated_2_sp_file_dir = os.listdir(comp_dir + folder + '/' + new_folder + '/' + sp_dir) 
            for new_spec in updated_2_sp_file_dir:
                if new_spec.endswith('.bin'):
                    pass
                else:
                    os.remove(comp_dir + folder + '/' + new_folder + '/' + sp_dir + '/' + new_spec)
            
            
# run the SpectralFilesJPGs through the meteor classifier model
            path = comp_dir + folder + '/' + new_folder + '/' + new_sp_dir
            names = os.listdir(path)
            names_array = []
            new_images = []
            for name in names:
                names_array.append(name)
            #converting images to np arrays and adding to testing set
            images = Path(path).glob('*.jpg') #finds images
            for image in images:
                img = Image.open(image)                    #opens as image object (str)
                pic = np.asarray(img)                      #converts to np.array
                pic = np.dot(pic[...,:3],[0.299, 0.587, 0.114])
                new_images.append(pic)

            # # normalize the data
            new_images = (np.asarray(new_images)) / 255

                                        
            model = keras.models.load_model('C:\CAMS\Runfolder\meteor_classification.h5')                           


            ### Check to see our accurate our predictions are
            predictions = model.predict(new_images)

            FalseSpectral = 'FalseSpectral'
            no_data_path = os.path.join(comp_dir + folder + '/' + new_folder , FalseSpectral)   
            os.makedirs(no_data_path, exist_ok=True)
            Meteors = 'Meteors'
            yes_data_path = os.path.join(comp_dir + folder + '/' + new_folder, Meteors)   
            os.makedirs(yes_data_path, exist_ok=True)
            count = 0


            for q in range(len(new_images)):
                try :
                    if predictions[q] <= 0.5:
                        #nonmeteor_data = pd.DataFrame({'Prediction': [predictions[q]],'Filename': [names_array[q]]})
                        #df_no = pd.concat([df_no, nonmeteor_data], ignore_index = True, axis=0)
                        shutil.move(comp_dir + folder + '/' + new_folder + '/' + new_sp_dir + '/' + names_array[q], comp_dir + folder + '/' + new_folder + '/' + FalseSpectral + '/' + names_array[q])
                    else:
                        #meteor_data = pd.DataFrame({'Prediction': [predictions[q]],'Filename': [names_array[q]]})
                        #df_yes = pd.concat([df_yes, meteor_data], ignore_index = True, axis=0)
                        shutil.move(comp_dir + folder + '/' + new_folder + '/' + new_sp_dir + '/' + names_array[q], comp_dir + folder + '/' + new_folder + '/' + Meteors + '/' + names_array[q])



                except FileNotFoundError:
                    count = count + 1
                    continue
            print('Done with classifying')
                        
            #print(count)

#remove now unused SpectralFilesJPGs folder
                   
print('End program')
import sys
sys.exit()
                            
