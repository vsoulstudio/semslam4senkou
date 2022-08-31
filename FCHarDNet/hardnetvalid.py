import torch
import argparse
import os
import cv2
import sys
import numpy as np
from collections import OrderedDict
from models import get_model
from skimage import img_as_ubyte
import dingyi as dy


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
def decode_segmap(temp,label_colours):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]
#        print(r,',',g,',',b)
    # print('hi')
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
def init_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 19
    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return device, model

def process_img(img, size, device, model):
#    print("Read Input Image from : {}".format(img_path))

#    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[0], size[1]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)




# # 19
#     colors = [  # [  0,   0,   0],
#         [128, 64, 128],
#         [244, 35, 232],
#         [70, 70, 70],
#         [102, 102, 156],
#         [190, 153, 153],
#         [153, 153, 153],
#         [250, 170, 30],
#         [220, 220, 0],
#         [107, 142, 35],
#         [152, 251, 152],
#         [0, 130, 180],
#         [220, 20, 60],
#         [255, 0, 0],
#         [0, 0, 142],
#         [0, 0, 70],
#         [0, 60, 100],
#         [0, 80, 100],
#         [0, 0, 230],
#         [119, 11, 32],
#     ]








#21
    # colors = [  
    #     # [0, 0, 0],
    #     [255,255,255],
    #     [128, 0, 0],
    #     [0, 128, 0],
    #     # [128, 128, 0],
    #     [0,0,0],
    #     [0, 0, 128],
    #     # [128, 0, 128],
    #     [0,0,0],
    #     [0, 128, 128],
    #     [128, 128, 128],
    #     [64, 0, 0],
    #     [192, 0, 0],
    #     [64, 128, 0],
    #     [192, 128, 0],
    #     [64, 0, 128],
    #     [192, 0, 128],
    #     [64, 128, 128],
    #     [192, 128, 128],
    #     [0,0,0],
    #     # [0, 64, 0],
    #     [128, 64, 0],
    #     [0, 192, 0],
    #     [128, 192, 0],
    #     [0, 64, 128],
    # ]


#19-0/255
    colors = [
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [0,0,0],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
    ]








#21-0/255
    # colors = [  
    #     [255,255,255],
    #     [255,255,255], 
    #     [255,255,255],
    #     # [128, 128, 0],
    #     [255,255,255],
    #     [255,255,255],
    #     # [128, 0, 128],
    #     [0,0,0],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     # [192, 128, 128],
    #     [0,0,0],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    #     [255,255,255],
    # ]

    label_colours = dict(zip(range(19), colors))
#    print (label_colours)
#    print('Output shape: ',outputs.shape)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
#    print (pred)
#    np.savetxt("result.txt", pred);
    # ~ with open('predsave.txt','w') as predsave:
        # ~ predsave.write(pred) 
    decoded = decode_segmap(temp=pred,label_colours=label_colours)

    return img_resized, decoded


device,model = init_model("pretrained/hardnet70_cityscapes_model.pkl")

def main(dpath):
    # fpath='/home/vsoul/bm_dataset/tum/rgbd_dataset_freiburg3_walking_rpy/rgb/'
    # foutpath='/home/vsoul/bm_dataset/tum/rgbd_dataset_freiburg3_walking_rpy/sego/'
    fpath = os.path.join(dpath+'/rgb/')
    foutpath = os.path.join(dpath+'/O4T1/')
    # print (foutpath2)
    if not (os.path.exists(foutpath)):
        os.makedirs(foutpath, mode=0o755)
    print(fpath)
    fnl=os.listdir(fpath)
    nb = len(fnl)
    a = 1
    for fn in fnl:
        img = cv2.imread(fpath+fn)
        sz=img.shape
        # print(sz)
        x,y = process_img(img,[sz[1],sz[0]],device,model.cuda())
        # print(y.shape)
        # # cv2.imshow("image2d",x)
        z = img_as_ubyte(y)
        # zz = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        # print(zz.shape)
        # ot = cv2.bitwise_and(img, img, mask = zz)
        # cv2.imwrite(foutpath+fn, ot)
        # cv2.imshow("image", ot)
        # cv2.waitKey(5)
        cv2.imwrite(foutpath+fn, z)
        print('\rLeft: ',nb-a, end = ' ')
        a=a+1
    print('1')
    return 0



if __name__ == '__main__':
    
    main(sys.argv[1])
    # main()
