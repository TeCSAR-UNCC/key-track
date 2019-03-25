import numpy as np



def load_openpose_features_train(w_img,h_img,video_frame_features,batch_size, num_steps, id):    
        
        video_frame_features = np.asarray(video_frame_features,dtype=np.float32)
        wid = w_img
        ht = h_img
        num_obj=1

        openpose_features=[]
        for frame_num in range(id,id+num_steps*batch_size):  # if id=2 gather frames from 2 to 7
            '''
            #Normalise the feature's x,y,
            temp = np.reshape(video_frame_features[frame_num][num_obj-1],newshape=(int(video_frame_features[frame_num][num_obj-1].shape[0]/3),-1))
            temp[:,0] = (temp[:,0])/wid
            temp[:,1] = (temp[:,1])/ht
            temp = np.reshape(temp,newshape=(54))
            '''
            opFeatures = video_frame_features[frame_num][num_obj-1].tolist()  #frame_num,obj1,255 features
            openpose_features =openpose_features + opFeatures
        openpose_features = np.asarray(openpose_features,dtype=np.float32)
        openpose_features = np.reshape(openpose_features, [batch_size*num_steps,54])
        
        return openpose_features

def load_openpose_gt_train(w_img,h_img,video_frame_bboxes, batch_size, num_steps, id):
    video_frame_bboxes = np.asarray(video_frame_bboxes,dtype=np.float32)
    num_obj=1
    wid = w_img
    ht = h_img
    openpose_gt=[]
    
    for frame_num in range(id+num_steps,id+1+(num_steps*batch_size),num_steps):
        video_frame_bboxes[frame_num][num_obj-1][0] += (video_frame_bboxes[frame_num][num_obj-1][2]/2.0) # convert x,y to centroid 
        video_frame_bboxes[frame_num][num_obj-1][1] += (video_frame_bboxes[frame_num][num_obj-1][3]/2.0)

        video_frame_bboxes[frame_num][num_obj-1][0] = (video_frame_bboxes[frame_num][num_obj-1][0]/wid)
        video_frame_bboxes[frame_num][num_obj-1][1] = (video_frame_bboxes[frame_num][num_obj-1][1]/ht)
        video_frame_bboxes[frame_num][num_obj-1][2] = (video_frame_bboxes[frame_num][num_obj-1][2]/wid)
        video_frame_bboxes[frame_num][num_obj-1][3] = (video_frame_bboxes[frame_num][num_obj-1][3]/ht)

        #print video_frame_bboxes[frame_num][num_obj-1]
        yoloBB = video_frame_bboxes[frame_num][num_obj-1].tolist() #frame_num, obj1,4 bbox attributes
        openpose_gt = openpose_gt + yoloBB
        openpose_gt = np.asarray(openpose_gt,dtype=np.float32)
    return openpose_gt

def iou(box1,box2):
     
    box_1=[box1[0],box1[1],box1[0]+box1[2],box1[1]+box1[3]]
    box_2=[box2[0],box2[1],box2[0]+box2[2],box2[1]+box2[3]]        
    b1_x0, b1_y0, b1_x1, b1_y1 = box_1
    b2_x0, b2_y0, b2_x1, b2_y1 = box_2
    
    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    if ((int_y1 <= int_y0) or (int_x1<=int_x0)):
        iou = 0

    else:
        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        # we add small epsilon of 1e-05 to avoid division by 0
        iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou 

def post_process_lstmout(gtarray,outputarray,w_img,h_img):
    '''
    Expects array of shape (obj/batch_size,4) for gt and output, both
    '''
    for obj in range(gtarray.shape[0]):
        if np.all(gtarray[obj]==0):
            outputarray[obj]=0
            continue
        gtarray[obj][0] = gtarray[obj][0] * w_img
        gtarray[obj][1] = gtarray[obj][1] * h_img
        gtarray[obj][2] = gtarray[obj][2] * w_img
        gtarray[obj][3] = gtarray[obj][3] * h_img        

        gtarray[obj][0] = gtarray[obj][0] - (gtarray[obj][2]/2.0) #Convert centroid X to left top X
        gtarray[obj][1] = gtarray[obj][1] - (gtarray[obj][3]/2.0) #Convert centroid Y to left top Y
        
        outputarray[obj][0] = outputarray[obj][0] * w_img
        outputarray[obj][1] = outputarray[obj][1] * h_img
        outputarray[obj][2] = outputarray[obj][2] * w_img
        outputarray[obj][3] = outputarray[obj][3] * h_img        

        outputarray[obj][0] = outputarray[obj][0] - (outputarray[obj][2]/2.0) #Convert centroid X to left top X
        outputarray[obj][1] = outputarray[obj][1] - (outputarray[obj][3]/2.0) #Convert centroid Y to left top Y
        

    return gtarray,outputarray

###################################################################################################################################

def load_openpose_features_MOT(w_img,h_img,video_frame_features,batch_size, num_steps, id):    
        
        video_frame_features = np.asarray(video_frame_features,dtype=np.float32)
        wid = w_img
        ht = h_img
        num_obj=batch_size

        openpose_features=[]
        for frame_num in range(id,id+num_steps):  # if id=2 gather frames from 2 to 7
            '''
            #Normalise the feature's x,y,
            temp = np.reshape(video_frame_features[frame_num][num_obj-1],newshape=(int(video_frame_features[frame_num][num_obj-1].shape[0]/3),-1))
            temp[:,0] = (temp[:,0])/wid
            temp[:,1] = (temp[:,1])/ht
            temp = np.reshape(temp,newshape=(54))
            '''
            optemp=[]
            for obj in range(num_obj):
                opFeatures = video_frame_features[frame_num][obj].tolist()  #frame_num,obj1,255 features
                optemp =optemp + opFeatures
            openpose_features = openpose_features+optemp 
        openpose_features = np.asarray(openpose_features,dtype=np.float32)
        openpose_features = np.reshape(openpose_features, [batch_size,num_steps,54])
        
        return openpose_features


def load_openpose_gt_MOT(w_img,h_img,video_frame_bboxes, batch_size, num_steps, id):
    video_frame_bboxes = np.asarray(video_frame_bboxes,dtype=np.float32)
    num_obj=batch_size
    wid = w_img
    ht = h_img
    openpose_gt=[]
    
    for frame_num in range(id+num_steps,id+1+(num_steps),num_steps):
        optemp=[]
        for obj in range(num_obj):
            video_frame_bboxes[frame_num][obj][0] += (video_frame_bboxes[frame_num][obj][2]/2.0) # convert x,y to centroid 
            video_frame_bboxes[frame_num][obj][1] += (video_frame_bboxes[frame_num][obj][3]/2.0)

            video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)
            video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)
            video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)
            video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)
        #print video_frame_bboxes[frame_num][num_obj-1]
            yoloBB = video_frame_bboxes[frame_num][obj].tolist() #frame_num, obj1,4 bbox attributes
            optemp = optemp + yoloBB
        openpose_gt = openpose_gt + optemp
    openpose_gt = np.asarray(openpose_gt,dtype=np.float32)
    openpose_gt = np.reshape(openpose_gt,[batch_size,4])
    return openpose_gt

##################################################################################################BATCH

def load_openpose_gt_batch(w_img,h_img,video_frame_bboxes, batch_size, num_steps, id):
    video_frame_bboxes = np.asarray(video_frame_bboxes,dtype=np.float32)
    num_obj=batch_size
    wid = w_img
    ht = h_img
    openpose_gt=[]
    
    for frame_num in range(id+num_steps,id+1+(num_steps),num_steps):
        optemp=[]
        for obj in range(num_obj):
            video_frame_bboxes[obj][frame_num][0] += (video_frame_bboxes[obj][frame_num][2]/2.0) # convert x,y to centroid 
            video_frame_bboxes[obj][frame_num][1] += (video_frame_bboxes[obj][frame_num][3]/2.0)

            video_frame_bboxes[obj][frame_num][0] = (video_frame_bboxes[obj][frame_num][0]/wid)
            video_frame_bboxes[obj][frame_num][1] = (video_frame_bboxes[obj][frame_num][1]/ht)
            video_frame_bboxes[obj][frame_num][2] = (video_frame_bboxes[obj][frame_num][2]/wid)
            video_frame_bboxes[obj][frame_num][3] = (video_frame_bboxes[obj][frame_num][3]/ht)
        #print video_frame_bboxes[frame_num][num_obj-1]
            yoloBB = video_frame_bboxes[obj][frame_num].tolist() #frame_num, obj1,4 bbox attributes
            optemp = optemp + yoloBB
        openpose_gt = openpose_gt + optemp
    openpose_gt = np.asarray(openpose_gt,dtype=np.float32)
    openpose_gt = np.reshape(openpose_gt,[batch_size,4])
    return openpose_gt

def load_openpose_features_batch(w_img,h_img,video_frame_features,batch_size, num_steps, id):
    video_frame_features = np.asarray(video_frame_features,dtype=np.float32)
    return video_frame_features[:batch_size,id:id+num_steps,:]

def iou_MOT(box1,box2):
    '''
    box1: 2D tensor of shape (frames,4 i.e[x,y,w,h])
    box2: 2D tensor of shape (frames,4 i.e[x,y,w,h])    
    ''' 
    vid_iou=0
    sub_count=0
    for i in range(box1.shape[0]):     
        #checks if the object is not present in some frames
        #hence counts the num of frames in which obj is not present
        if np.all(box1[i]==0):  
            sub_count+=1

        box_1=[box1[i][0],box1[i][1],box1[i][0]+box1[i][2],box1[i][1]+box1[i][3]]
        box_2=[box2[i][0],box2[i][1],box2[i][0]+box2[i][2],box2[i][1]+box2[i][3]]        
        b1_x0, b1_y0, b1_x1, b1_y1 = box_1
        b2_x0, b2_y0, b2_x1, b2_y1 = box_2
        
        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        if ((int_y1 <= int_y0) or (int_x1<=int_x0)):
            iou = 0

        else:
            int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

            b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
            b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

            # we add small epsilon of 1e-05 to avoid division by 0
            iou = int_area / (b1_area + b2_area - int_area + 1e-05)
        vid_iou=vid_iou+iou
    if box1.shape[0]-sub_count==0:  #For those batches where object is not at all present
        return 0
    else:
        return vid_iou/(box1.shape[0]-sub_count) #Subtract those number of frames where object is not present at all 
