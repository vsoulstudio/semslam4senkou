import os
import cv2
import numpy as np


def maskprocess(seg, depo):
    '''
    输入原始深度图和seg图，输出修正后的mask
    '''
    seg1 = 255 - seg
    dep = cv2.bitwise_and(depo, depo, mask = seg1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    msk  = cv2.dilate(seg1, kernel, iterations = 20)

    cts, hi = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(cts)):
        x, y, w, h = cv2.boundingRect(cts[i])
        msk[...,x:x+w+1] = 255


    # x, y, w, h = cv2.boundingRect(msk)
    # msk[...,x:x+w+1] = 255
    md2 = cv2.bitwise_and(depo, depo, mask = msk)
    h_10 = 0
    h_90 = 0
    if dep.max() > 0:
        h_10 = np.percentile(dep[dep > 0],10).astype(int)
        h_90 = np.percentile(dep[dep > 0],90).astype(int)
    h_10 = h_10 - 10
    h_90 = h_90 + 5
    md2[md2<h_10] = 0
    md2[md2>h_90] = 0
    md3 = md2
    md3[md3>0] = 255
    # md3 = 255 - md3
    # mdd = cv2.addWeighted(md3, 0.5, seg, 0.5, 0)
    # mdd[mdd>0] = 255
    mdd = cv2.add(md3, seg1)
    
    for i in range(0,640):
        if np.percentile(mdd[...,i],90) == 0:
            mdd[...,i]=0
    mdd = 255-mdd
    return mdd

def load_r_d(path_to_association):
    '''
    返回：rgbimg,depimg,rgbtime,deptime
    '''
    rgbf = []
    rgbt = []
    depf = []
    dept = []

    with open(os.path.join(path_to_association, 'associate.txt')) as times_file:
        for line in times_file:
            if len(line) > 0 and not line.startswith('#'):
                rt, rgb,dt,depth = line.rstrip().split(' ')[0:4]
                rgbf.append(rgb)
                # rgbt.append(float(rt))
                # dept.append(float(dt))
                rgbt.append(rt)
                dept.append(dt)
                depf.append(depth)
    return rgbf, depf, rgbt, dept

