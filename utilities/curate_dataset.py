cam_id = 8.0
which_split='testing'

import pandas as pd

import random

from scipy.io import loadmat
from shapely.geometry import MultiPoint,box

gt = loadmat('../mat_files/ground_truth/trainval.mat')
gt = gt['trainData']


import numpy as np
import h5py 

for cam_id in [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]:
    f = h5py.File('../mat_files/detections/OpenPose/camera'+str(cam_id)[0]+'.mat','r') 
    opose_det = f.get('detections')
    opose_det = np.array(opose_det)
    opose_det= np.transpose(opose_det)


    data = gt
    df_gt = pd.DataFrame(data,columns=['Camera','ID','frame','topX','topY','wid','ht','wX','wY','fX','fY'])

    del df_gt['wX']
    del df_gt['wY']
    del df_gt['fX']
    del df_gt['fY']

    grouped = df_gt.groupby('Camera')
    tempframe =grouped.get_group(cam_id)#.groupby('ID').get_group(obj_id)


    a = tempframe['ID'].unique()
    obj_ids = random.sample(list(a),8)

    #obj_ids = [3000.0]
    for obj_id in obj_ids:
        print('Cam: '+str(cam_id)+' Obj: '+str(obj_id))
        gt_dataframe=grouped.get_group(cam_id).groupby('ID').get_group(obj_id)


        opendata = opose_det #Whole detection 
        openpose_dataframe = pd.DataFrame(opendata,columns=['camera', 'frame', 'kpx1', 'kpy1', 'kpc1','kpx2', 'kpy2', 'kpc2','kpx3', 'kpy3', 'kpc3','kpx4', 'kpy4', 'kpc4','kpx5', 'kpy5', 'kpc5','kpx6', 'kpy6', 'kpc6','kpx7', 'kpy7', 'kpc7','kpx8', 'kpy8', 'kpc8','kpx9', 'kpy9', 'kpc9','kpx10', 'kpy10', 'kpc10','kpx11', 'kpy11', 'kpc11','kpx12', 'kpy12', 'kpc12','kpx13', 'kpy13', 'kpc13','kpx14', 'kpy14', 'kpc14','kpx15', 'kpy15', 'kpc15','kpx16', 'kpy16', 'kpc16','kpx17', 'kpy17', 'kpc17', 'kpx18', 'kpy18', 'kpc18'])



        output =[]
        output_gt = []
        for row_index_gt,row_gt in gt_dataframe.iterrows(): #[Camera,ID,frame,topX,topY,wid,ht]
            op_dataframe=openpose_dataframe.groupby('frame').get_group(row_gt[2])
            bbox = box((row_gt[3])/1920.0,(row_gt[4])/1080.0,(row_gt[3]+row_gt[5])/1920.0,(row_gt[4]+row_gt[6])/1080.0)
            #print (row_gt[2])
            for row_index_op,row_op in op_dataframe.iterrows(): #[camera,frame,kpx1,kpy1,kpc1,kpx2,kpy2,kpc2,kpx3,kpy3,...,kpc15,kpx16,kpy16,kpc16,kpx17,kpy17,kpc17,kpx18,kpy18,kpc18]
                
                kpoints = MultiPoint([
                    (row_op[2],row_op[3]),
                    (row_op[5],row_op[6]),
                    (row_op[8],row_op[9]),
                    (row_op[11],row_op[12]),
                    (row_op[14],row_op[15]),
                    (row_op[17],row_op[18]),
                    (row_op[20],row_op[21]),
                    (row_op[23],row_op[24]),
                    (row_op[26],row_op[27]),
                    (row_op[29],row_op[30]),
                    (row_op[32],row_op[33]),
                    (row_op[35],row_op[36]),
                    (row_op[38],row_op[39]),
                    (row_op[41],row_op[42]),
                    (row_op[44],row_op[45]),
                    (row_op[47],row_op[48]),
                    (row_op[50],row_op[51]),
                    (row_op[53],row_op[54]),
                ])
                cnt=0
                for i in range(18):
                    if bbox.contains(kpoints[i]):
                        cnt =cnt+1
                if cnt>8:
                    #print (row_gt)
                    output.append(pd.np.array(row_op[2:],dtype='float64'))
                    output_gt.append(pd.np.array(row_gt[3:],dtype='float64'))
                    break
                    


        # In[40]:


        final = np.asarray(output,dtype=np.float64)
        final =np.reshape(final,newshape=(final.shape[0],1,final.shape[1]))

        final_gt = np.asarray(output_gt,dtype=np.float64)
        final_gt =np.reshape(final_gt,newshape=(final_gt.shape[0],1,final_gt.shape[1]))


        print('The frames for ',obj_id,' is ',final.shape,final_gt.shape)


        # In[42]:

        if which_split=='training':
            np.save('duke_dataset/training/'+'cam'+str(cam_id)[:-2]+'obj'+str(obj_id)[:-2],final)
            np.save('duke_dataset/training/'+'cam'+str(cam_id)[:-2]+'obj'+str(obj_id)[:-2]+'_gt',final_gt)
        elif which_split=='testing':
            np.save('duke_dataset/testing_scaling/'+'cam'+str(cam_id)[:-2]+'obj'+str(obj_id)[:-2],final)
            np.save('duke_dataset/testing_scaling/'+'cam'+str(cam_id)[:-2]+'obj'+str(obj_id)[:-2]+'_gt',final_gt)