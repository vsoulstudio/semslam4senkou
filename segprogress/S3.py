import os
import dingyi as dy
import sys
import time

def rw(path):
    rgbf, depf, rgbt, dept = dy.load_r_d(path)
    num = len(rgbt)
    xr = open(os.path.join(path,'newassociate.txt'),"w")
    for i in range(0,num):
        segstr = 'seg/'+rgbt[i]+'.png'
        st = rgbt[i] +" "+rgbf[i]+' '+dept[i]+' '+depf[i]+' '+segstr+'\n'
        # print (st)
        xr.write(st)
    xr.close()

    return 0
    
def main(path):
    stt = time.time()
    rw(path)
    edt = time.time()
    print(edt-stt)
    return 0

if __name__ == '__main__':
    
    main(sys.argv[1])
