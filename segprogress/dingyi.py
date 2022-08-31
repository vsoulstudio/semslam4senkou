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
        h_10 = np.percentile(dep[dep > 0],40).astype(int)
        h_90 = np.percentile(dep[dep > 0],60).astype(int)
    h_10 = h_10 - 2
    h_90 = h_90 + 5
    md2[md2<h_10] = 0
    md2[md2>h_90] = 0
    md3 = md2
    md3[md3>0] = 255
    # md3 = 255 - md3
    # mdd = cv2.addWeighted(md3, 0.5, seg, 0.5, 0)
    # mdd[mdd>0] = 255
    mdd = cv2.add(md3, seg1)
    
    # for i in range(0,640):
    #     if np.percentile(mdd[...,i],90) == 0:
    #         mdd[...,i]=0
    mdd = 255-mdd
    return mdd

def maskprocess_v2(seg, depo, dp_fr, dp_af):
    '''
    输入原始深度图,seg图,上次的中心深度，下次的中心深度
    输出修正后的mask，这次的中心深度, 是否需要补齐
    '''

#判断 如果全白且处于识别到的队列头、尾则跳过，如果全白且前后都有识别到则退出该部分返回true进入补偿阶段
    if seg.min() == 255 and (dp_fr == 0 or dp_af == 0):
        return seg, 0, False
    elif seg.min() == 255 and (dp_fr != 0 and dp_af != 0):
        return seg, 0, True

#扩大mask范围
    seg1 = 255 - seg
    dep = cv2.bitwise_and(depo, depo, mask = seg1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    msk  = cv2.dilate(seg1, kernel, iterations = 20)

    cts, hi = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(cts)):
        x, y, w, h = cv2.boundingRect(cts[i])
        msk[...,x:x+w+1] = 255

#在depth图上应用mask提取主要部分作为人来识别
    # x, y, w, h = cv2.boundingRect(msk)
    # msk[...,x:x+w+1] = 255
    md2 = cv2.bitwise_and(depo, depo, mask = msk)
    h_10 = 0
    h_90 = 0
    if dep.max() > 0:
        h_10 = np.percentile(dep[dep > 0],40).astype(int)
        h_90 = np.percentile(dep[dep > 0],60).astype(int)
    # h_10 = np.percentile(dep[dep > 0],40).astype(int)
    # h_90 = np.percentile(dep[dep > 0],60).astype(int)
    h_10 = h_10 - 10
    h_90 = h_90 + 10
    
    md3 = md2
    md3[md3<h_10] = 0
    md3[md3>h_90] = 0
    md3[md3>0] = 255
    
    
#计算depth中值
    od = cv2.bitwise_and(depo, depo, mask = md3)
    n_dpth = 0
    if od.max() > 0:
        n_dpth = np.percentile(od[od>0], 50).astype(int) 
#如果该frame的depth不在前后depth中值的区间内则设定为中值
    if dp_fr != 0 and dp_af != 0:
        dp_fa = np.array([dp_fr, dp_af])
        
        if n_dpth < dp_fa.min()-5 or n_dpth > dp_fa.max()+5:
            n_dpth = dp_fa.mean().astype(int)
            nu = n_dpth + 15
            nd = n_dpth - 15
            md4 = md2
            md4[md4 < nd] = 0
            md4[md4 > nu] = 0
            md4[md4 > 0] = 255
            md4 = 255 - md4
            return md4, n_dpth, False


    # mdd = cv2.add(md3, seg1)
    # for i in range(0,640):
    #     if np.percentile(mdd[...,i],90) == 0:
    #         mdd[...,i]=0

#翻转黑白 输出mask
    # mdd = 255 - mdd
    md3 = 255 - md3
    
    return md3, n_dpth, False

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

def load_rds(path_to_association):
    '''
    返回：rgbimg,depimg,rgbtime,deptime
    '''
    rgbf = []
    rgbt = []
    depf = []
    dept = []
    seg = []

    with open(os.path.join(path_to_association, 'newassociate.txt')) as times_file:
        for line in times_file:
            if len(line) > 0 and not line.startswith('#'):
                rt, rgb, dt, depth, se = line.rstrip().split(' ')[0:5]
                rgbf.append(rgb)
                # rgbt.append(float(rt))
                # dept.append(float(dt))
                rgbt.append(rt)
                dept.append(dt)
                depf.append(depth)
                seg.append(se)
    return rgbf, depf, rgbt, dept, seg

def main(datapath):
    rp, dp, rt, dt  = dy.load_r_d(datapath)
    rp, dp, rt, dt, se = dy.loadrds(datapath)
    opth = os.path.join(datapath+'out4test/')
    nums = len(rt)
    if not (os.path.exists(opth)):
        os.makedirs(opth, mode=0o777)
        print('folder created')
    for i in range (0,nums):
        rgb = cv2.imread(os.path.join(datapath+rp[i]), cv2.IMREAD_UNCHANGED)
        dep = cv2.imread(os.path.join(datapath+dp[i]), cv2.IMREAD_UNCHANGED)
        seg = cv2.imread(os.path.join(datapath+se[i]), cv2.IMREAD_GRAYSCALE)

    return 0

if __name__ == '__main__':
    
    main(sys.argv[1])
