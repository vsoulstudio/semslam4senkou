#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test.py
#  
#  Copyright 2020 vsoul <vsoul@vsoul>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import numpy as np
import os
import sys
import cv2
import dingyi as dy
import time

def main(sequence_path):
    # sequence_path = '/home/vsoul/bm_dataset/tum/rgbd_dataset_freiburg3_walking_halfsphere/'
    rp, dp, rt, dt  = dy.load_r_d(sequence_path)
    num_images = len(rt)
    weiba = '.png'
    fop1 = os.path.join(sequence_path+'/repair1st')
    fop2 = os.path.join(sequence_path+ '/repair2nd')
    fip = os.path.join(sequence_path+ '/sego/')
    if not (os.path.exists(fop1)):
        os.makedirs(fop1, mode=0o777)
    if not (os.path.exists(fop2)):
        os.makedirs(fop2, mode=0o777)
    print('image nums: ', num_images)
    
    print('round 1')    
    # print(os.path.join(fip, rt[0]+ weiba))
    image_1 = cv2.imread(os.path.join(fip, rt[0]+ weiba), cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(os.path.join(fip, rt[1]+ weiba), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(fop1, rt[0]+ weiba), image_1)
    cv2.imwrite(os.path.join(fop1, rt[1]+ weiba), image_2)
    image_3 = cv2.imread(os.path.join(fip, rt[2]+ weiba), cv2.IMREAD_GRAYSCALE)
    image_4 = cv2.imread(os.path.join(fip, rt[3]+ weiba), cv2.IMREAD_GRAYSCALE)


    ts = time.time()
    for idx in range( 4, num_images):
        tsc = time.clock()
        image_5 = cv2.imread(os.path.join(fip, rt[idx]+ weiba), cv2.IMREAD_GRAYSCALE)
        imgo = ctn(image_1, image_2, image_3, image_4, image_5)
        tec = time.clock()
        cv2.imwrite(os.path.join(fop1, rt[idx - 2]+ weiba), imgo)
        print('\rFileName', os.path.join(rt[idx - 2]+ weiba), '  Last: ', num_images - idx -1,  'cpu time', tec-tsc, end = ' ')
        image_1 = image_2
        image_2 = image_3
        image_3 = image_4
        image_4 = image_5
    te = time.time()
    print(ts,'\n', te,'\n', te-ts,'\n', (te-ts)/num_images)
    # return 0

    cv2.imwrite(os.path.join(fop1, rt[num_images-2]+ weiba), image_3)
    cv2.imwrite(os.path.join(fop1, rt[num_images-1]+ weiba), image_4)

    print ('\nround 2')
    image_1 = cv2.imread(os.path.join(fop1, rt[0]+ weiba), cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(os.path.join(fop1, rt[1]+ weiba), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(fop2, rt[0]+ weiba), image_1)
    cv2.imwrite(os.path.join(fop2, rt[1]+ weiba), image_2)
    image_3 = cv2.imread(os.path.join(fop1, rt[2]+ weiba), cv2.IMREAD_GRAYSCALE)
    image_4 = cv2.imread(os.path.join(fop1, rt[3]+ weiba), cv2.IMREAD_GRAYSCALE)
    for idx in range( 4, num_images):
        image_5 = cv2.imread(os.path.join(fop1, rt[idx]+ weiba), cv2.IMREAD_GRAYSCALE)
        imgo = ctn(image_1, image_2, image_3, image_4, image_5)
        cv2.imwrite(os.path.join(fop2, rt[idx - 2]+ weiba), imgo)
        print('\rFileName', os.path.join(rt[idx - 2]+ weiba), '  Last: ', num_images - idx -1, end = ' ')
        image_1 = image_2
        image_2 = image_3
        image_3 = image_4
        image_4 = image_5
    cv2.imwrite(os.path.join(fop2, rt[num_images-2]+ weiba), image_3)
    cv2.imwrite(os.path.join(fop2, rt[num_images-1]+ weiba), image_4)

    print('Finished\n')


    return 0

def ctn(img1, img2, img3, img4, img5 ):
    cr13 = img1 < img3
    cr23 = img2 < img3
    cr43 = img4 < img3
    cr53 = img5 < img4
    cp12 = np.logical_or(cr13,cr23)
    cp45 = np.logical_or(cr43,cr53)
    cpr = np.logical_and(cp12,cp45)
    img = img3 - cpr*255
    return  img

def load_images(path_to_association):
    rgb_fn = []
    ts = []
    with open(os.path.join(path_to_association, 'rgb.txt')) as times_file:
        for line in times_file:
            if len(line) > 0 and not line.startswith('#'):
                t, rgb = line.rstrip().split(' ')[0:2]
                rgb_fn.append(rgb)
                ts.append(float(t))
    return rgb_fn, ts
    
if __name__ == '__main__':
    
    main(sys.argv[1])
    # main()
