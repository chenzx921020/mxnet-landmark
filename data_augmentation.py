# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os,sys
import cv2
import numpy as np
import math
from skimage import exposure


def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def rotate_about_lmk(src,angle):
    rotate_mat = [np.cos(angle),-1*np.sin(angle),np.sin(angle),np.cos(angle)]
    rotate_mat = np.array(rotate_mat).reshape(2,2)
    dst = np.dot(src,rotate_mat)
    return dst

root_dir = '/data/zhixuan/lmk_data/crop_data/crop_face_train/'
lst_path = '/data/zhixuan/lmk_data/crop_data/crop_re_list/train.lst'
new_lst = '/data/zhixuan/lmk_data/crop_data/crop_re_list/aug.lst'
f=open(new_lst,'wb')
lines = open(lst_path).readlines()
k=len(lines)
for line in lines:
    
    tmp = line.split()
    img_url = tmp[-1]
    img_path = root_dir + img_url
    img_data = cv2.imread(img_path)
    lmk_data = []
    for i in range(0,42):
        lmk_data.append(float(tmp[i+1]))
    lmk_data=np.array(lmk_data).reshape(-1,2)
    
    width,height,c=img_data.shape
    # rotated augmentation
    angle = np.random.randint(0,60)
    print angle
    
    
    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)  
    rotateImg = cv2.warpAffine(img_data, rotateMat, (width, height))  
    #rotated_img = rotate_about_center(img_data,angle)
    cx = width/2
    cy = height/2
    rot_data=np.zeros(42).reshape(-1,2)
    rot_data[:,0]=lmk_data[:,0]-cx
    rot_data[:,1]=lmk_data[:,1]-cy
    rot_data = rotate_about_lmk(rot_data,(np.pi*angle)/180)
    rot_data[:,0]=rot_data[:,0]+cx
    rot_data[:,1]=rot_data[:,1]+cy
    
#    for i in range(0,21):
#        cv2.circle(rotateImg,(int(rot_data[i,0]),int(rot_data[i,1])),2,(0,255,0))
#    cv2.imshow('test',rotateImg)
#    cv2.waitKey(0)
            
    f.write(str(k)+'\t')
    for j in range(0,21):
        f.write(str(rot_data[j,0]) + '\t' + str(rot_data[j,1]) + '\t')
    f.write(img_url[:-4]+'_rot.png' +'\n')
    cv2.imwrite(root_dir+img_url[:-4]+'_rot.png',rotateImg)
    k=k+1
    

    # illumination augmentation    
    gamma1 = float(np.random.randint(1,19))/10
    #gamma2 = float(np.random.randint(1,9))/10 
    print gamma1
    illu_img= exposure.adjust_gamma(img_data, gamma1)
    #bright_img = exposure.adjust_gamma(img_data, gamma2) #bright
#    cv2.imshow('illu_img',illu_img)
#    cv2.waitKey(0)
    f.write(str(k)+'\t')
    for j in range(0,21):
        f.write(str(lmk_data[j,0]) + '\t' + str(lmk_data[j,1]) + '\t')
    f.write(img_url[:-4]+'_illu.png' +'\n')
    cv2.imwrite(root_dir+img_url[:-4]+'_illu.png',illu_img)
    k=k+1

    #flip
    direction = np.random.randint(0,2)
    flip_img = cv2.flip(img_data,direction)
    flip_data=np.zeros(42).reshape(-1,2)    
    flip_data[:,0]=lmk_data[:,0]-cx
    flip_data[:,1]=lmk_data[:,1]-cy
    if direction==1:
        flip_data[:,0]=-1*flip_data[:,0]
    else:
        flip_data[:,1]=-1*flip_data[:,1]
    flip_data[:,0]=flip_data[:,0]+cx
    flip_data[:,1]=flip_data[:,1]+cy
                                                                         
    f.write(str(k)+'\t')
    for j in range(0,21):
        f.write(str(flip_data[j,0]) + '\t' + str(flip_data[j,1]) + '\t')
    f.write(img_url[:-4]+'_flip.png' +'\n')
    cv2.imwrite(root_dir+img_url[:-4]+'_flip.png',flip_img)
    #    for i in range(0,21):
    #        cv2.circle(flip_img,(int(flip_data[i,0]),int(flip_data[i,1])),2,(0,255,0))
    #    cv2.imshow('flip_data',flip_img)
    #    cv2.waitKey(0)
    k=k+1
    
    # random erasing
    erase_x1=np.random.randint(0,width)
    erase_x2=np.random.randint(erase_x1,width)
    erase_y1=np.random.randint(0,height)
    erase_y2=np.random.randint(erase_y1,height)
    erase_img = img_data
    erase_img[erase_x1:erase_x2,erase_y1:erase_y2]=erase_img[erase_x1:erase_x2,erase_y1:erase_y2]+np.mean(erase_img)
#    cv2.imshow('erase_img',erase_img)
#    cv2.waitKey(0)
    f.write(str(k)+'\t')
    for j in range(0,21):
        f.write(str(lmk_data[j,0]) + '\t' + str(lmk_data[j,1]) + '\t')
    f.write(img_url[:-4]+'_erase.png' +'\n')
    cv2.imwrite(root_dir+img_url[:-4]+'_erase.png',erase_img)
    k=k+1
f.close()    
        

    
    
    
