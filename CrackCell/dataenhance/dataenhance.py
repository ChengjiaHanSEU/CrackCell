import numpy as np
import os
import cv2
    
def put_mask(img,mask,output_fold):

    # 1.读取图片
    image = np.uint8(img)
    mask = cv2.resize(mask, image.shape[::-1][1:])

    mask = np.tile(mask[..., np.newaxis], (1,1,3))

    # 3.画出mask
    zeros_mask = mask
        # alpha 为第一张图片的透明度
    alpha = 1
        # beta 为第二张图片的透明度 
    beta = -np.random.uniform(0.1, 0.4)
    gamma = 0
    
        # cv2.addWeighted 将原始图片与 mask 融合
    mask_img =  cv2.addWeighted(image, alpha, zeros_mask, beta, gamma)
    mask_img = cv2.cvtColor(mask_img,cv2.COLOR_RGB2GRAY)
    
    mask_img = mask_img + np.random.randint(-20, 20)
    
    return mask_img


def getmask(path='./mask/'):

    choice_list = ['tree', 'afflicted']
    choice_dir = np.random.choice(choice_list)
    listofmask = os.listdir(path+'/'+choice_dir)
    choice_mask = np.random.choice(listofmask)
    maskpath = path+'/'+choice_dir+'/'+choice_mask
    mask = cv2.imread(maskpath, 0)
    mask = np.uint8(mask)
    mask = 255 - mask
    mask[mask>5] = 255
    mask[mask<=5] = 0

    return mask

def imgpreprocess(img):

    height,width=img.shape[:2]
    #镜像
    flip_num = np.random.choice([0, 1, 2, 3])
    img=cv2.flip(img, flip_num)
    #旋转
    center = (width // 2, height // 2)
    rot_num = np.random.choice(np.linspace(360,10))
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    img = cv2.warpAffine(img, M, (height,width))
    
    #仿射变换
    point1=np.float32([[0,0],\
                       [0,height],\
                     [ width,height]])
    point2=np.float32([[0,0],\
                       [0,np.random.randint(0, height)],\
                     [np.random.randint(0, width),np.random.randint(0, height)]])
    M=cv2.getAffineTransform(point1,point2)
    img=cv2.warpAffine(img,M, (height,width))

    #缩放
    scale_factor = np.random.uniform(1, 5)
    img=cv2.resize(img,(int(scale_factor*width),int(scale_factor*height)))
    
    #平移
    M=np.array([[1,0,np.random.randint(-width/8, width/8)],\
                [0,1,np.random.randint(-height/8, height/8)]],\
               dtype=np.float32)
    img=cv2.warpAffine(img,M, (height,width))
    
    indexgauss = np.random.randint(0,2)
    
    if indexgauss == 0:
    
      img= cv2.GaussianBlur(img, (31, 31), 31)
    else:
      img= cv2.GaussianBlur(img, (11, 11), 11)
    
    return img


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask
def generate_stroke_mask(im_size=(640,640), max_parts=15, maxVertex=25, maxLength=120, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = np.random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    #mask = np.concatenate([mask, mask, mask], axis = 2)
    mask[mask>0] = 255
    #mask[mask<=0] = 0
    mask = np.uint8(mask)
    return mask