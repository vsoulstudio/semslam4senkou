import os
import numpy as np
import cv2
import sys
import dingyi as dy
import time

def main(datapath):
    rp, dp, rt, dt = dy.load_r_d(datapath)
    nums = len(rt)
    # dfr = 0
    # daf = 0
    # dflag = False
    # llst = []
    opth = os.path.join(datapath+'/seg/')
    print (opth)
    if not (os.path.exists(opth)):
        os.makedirs(opth, mode=0o755)
    st = time.time()
    for idx in range(0, nums):
        
        deppth = os.path.join(datapath, dp[idx])
        segpth = os.path.join(datapath+'/repair2nd/', str(rt[idx])+'.png')
        rgbimg = os.path.join(datapath, rp[idx])
        savepth = os.path.join(datapath+ '/seg/', str(rt[idx])+'.png')
        
        daf = 0
        # if idx < nums-1:
        #     dep2 = cv2.imread(os.path.join(datapath, dp[idx+1]), cv2.IMREAD_GRAYSCALE)
        #     if dep2.max() > 0:
        #         daf = np.percentile(dep2[dep2 > 0], 50).astype(int)

        # print('segpth = ',segpth)
        # print('deppth = ',deppth)
        # print('rgbpth = ', rgbimg)
        # print('svpth = ', savepth)
        
        depo = cv2.imread(deppth, cv2.IMREAD_GRAYSCALE)
        sego = cv2.imread(segpth, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(rgbimg)
        stc= time.clock()
        # cv2.imshow('seg',sego)
        msk = dy.maskprocess(sego, depo)
        # msk, dnow, dflag = dy.maskprocess_v2(sego, depo, dfr, daf)
        # if dflag:
        #     llst.append(segpth)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # msk  = cv2.dilate(msk, kernel, iterations = 80)
        # msk = cv2.erode(msk, kernel, iterations= 5)
        # dfr = dnow
        # out = cv2.bitwise_and(rgb, rgb, mask = msk)
        # cv2.imshow('www',out)
        # cv2.waitKey(1)
        etc = time.clock()

        # print('msk: ',msk.shape)
        # print('out: ',out.shape)
        
        print('\rlast:', nums - idx -1, '     ' , etc - stc, end = ' ')

        cv2.imwrite(savepth, msk)
    et = time.time()
    tt = et - st
    at = tt / nums
    print (at)
    # print('picnums: ',len(llst))
    # for i in range (0, len(llst)):
    #     print (llst[i]) 
    print('Finished')
    return 0


if __name__ == '__main__':
    
    main(sys.argv[1])
