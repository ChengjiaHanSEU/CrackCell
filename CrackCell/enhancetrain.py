
import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Flatten,Reshape, Dense,Multiply,Conv2D, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, concatenate, Add, Activation
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def getdata(ipath, mpath):
    mask_path_list = []
    img_path_list = []
    for filename in os.listdir(ipath):
    
        img_path = ipath + '/' + filename
        img_path_list.append(img_path)
        
        mask_path = mpath + '/' + filename[:-4] + '.png'
        mask_path_list.append(mask_path)
        
    return img_path_list, mask_path_list 

import matplotlib.pyplot as plt

def my_conv(input_, kernel, s):
    '''plt.figure()
    plt.imshow(input_)
    plt.show()'''
    output_size_0 = int((len(input_) - len(kernel)) / s + 1)   
    output_size_1 = int((len(input_[0]) - len(kernel[0])) / s + 1)   
    res = np.zeros([output_size_0, output_size_1], np.float32)

    for i in range(len(res)):
        for j in range(len(res[0])):
            a = input_[i*s:i*s + len(kernel), j*s: j*s + len(kernel)] 
            b = a * kernel 
            res[i][j] = b.sum()
    res[res > 10] = 255
    res[res <= 10] = 0
    '''plt.figure()
    plt.imshow(res)
    plt.show()'''
    return res
from dataenhance.dataenhance import getmask, generate_stroke_mask, imgpreprocess, put_mask

def enhanceblock(img):
    func = [getmask, generate_stroke_mask]
    index = 0#np.random.randint(0, 2)
    mask = func[index]()

    if index == 0:
        mask = imgpreprocess(mask)

#mask = np.uint8(mask)
    outimg = put_mask(img,mask,output_fold='./')
    return outimg

def read_image(image_path, mask_path, resize_shape = (1200, 1200), grad_size = 100):
    kernel = np.ones((grad_size, grad_size))
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    image = cv2.resize(image, resize_shape)
    mask = cv2.resize(mask, resize_shape)
    
    '''plt.figure()
    plt.imshow(image)
    plt.show()
    
    plt.figure()
    plt.imshow(mask)
    plt.show()'''
    
    
    index= 1  # np.random.randint(0, 2)
    if index == 1:
        image = enhanceblock(image)
        
    else :
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    mask = my_conv(mask/255., kernel, grad_size)


    image = image[..., np.newaxis].astype('float32')
    mask = mask[..., np.newaxis].astype('float32')
    

    
    image = image / 255.
    mask = mask / 255.
    
    return image, mask

ken_size = (3, 3)     # 3, 5 , 7, 9, 11, 13, 15
ken_num = 8           #8s, 16l
each_layers = 1       #1, 2
dilation_rate = 1     # 1, 2, 3

def v_block(x):
    x1 = Conv2D(ken_num * 5, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x)
    x1 = Dropout(0.5)(x1)
    for i in range(each_layers):
      x1 = Conv2D(ken_num * 5, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x1)
      x1 = Dropout(0.5)(x1)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x2 = Conv2D(ken_num * 6, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x1)
    x2 = Dropout(0.5)(x2)
    x2 = Conv2D(ken_num * 8, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Conv2D(ken_num * 6, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x2)
    x2 = Dropout(0.5)(x2)
    x2 = UpSampling2D(2)(x2)
    x2 = concatenate([x, x2], axis=-1)   
    x3 = Conv2D(ken_num * 5, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x2)
    x3 = Dropout(0.5)(x3)
    for i in range(each_layers):
      x3 = Conv2D(ken_num * 3, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x3)   
      x3 = Dropout(0.5)(x3)
    return x3

def FCN(input_size = (1200, 1200, 1)):
    input_ = tf.keras.layers.Input(shape=input_size)
    x = Conv2D(ken_num, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(input_)
    for i in range(each_layers):
      x = Conv2D(ken_num, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate+ i)(x)
    x1 = MaxPooling2D((5, 5), padding='same')(x)
    
    x2 = Conv2D(ken_num * 2, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x1)
    for i in range(each_layers):
        x2 = Conv2D(ken_num * 2, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate+ i)(x2)
    x2 = MaxPooling2D((5, 5), padding='same')(x2)
    
    x3 = Conv2D(ken_num * 3, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x2)
    for i in range(each_layers):
      x3 = Conv2D(ken_num * 3, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate+ i)(x3)
    x3 = MaxPooling2D((2, 2), padding='same')(x3)
    
    x4 = Conv2D(ken_num * 4, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate)(x3)
    for i in range(each_layers):
        x4 = Conv2D(ken_num * 4, ken_size, activation='relu', padding='same', dilation_rate = dilation_rate+ i)(x4)
    x4 = MaxPooling2D((2, 2), padding='same')(x4)
    
    x4 = v_block(x4)
    out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x4)
    model = tf.keras.Model(inputs=input_, outputs=out)
    model.summary()
    return model
    


def gene_images(gen_model, poorimage, epoch):
    pred = gen_model.predict(poorimage)
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow((pred[i, :, :, 0] + 1) / 2, cmap='gray')
        plt.axis('off')
    fig.savefig("./images/crack_%d.png" % epoch)
    plt.close()

BCE = tf.keras.losses.BinaryCrossentropy()

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    lr = 0.001
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    epochs = 25
    
    batch_size = 4
    inputsize = (1200,1200)
    masksize = (12, 12)
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    path1, path2 = getdata(r'./dataset/data/imgs', r'./dataset/data/labels')
    
    
    num = np.arange(len(path1))
    
    np.random.shuffle(num)
    
    path1 = list(np.array(path1)[num])
    path2 = list(np.array(path2)[num])
    
    
    path1train = path1[:-100]
    path2train = path2[:-100]
    path1val = path1[-100:]
    path2val = path2[-100:]
    #print(path1)
    #read_image(path1[0], path2[0])
    SEGPAVER = FCN()
    
    #SEGPAVER = tf.keras.models.load_model('')      # segmentnet()

    min_loss = 1
    while epochs:
        indexofdata = 0

        while indexofdata < len(path1train):
            imageinput = np.empty((batch_size, inputsize[0], inputsize[1], 1))
            maskout = np.empty((batch_size, masksize[0], masksize[1], 1))
            for i in range(indexofdata, indexofdata + batch_size):
                imageinput[i - indexofdata, ...], \
                maskout[i- indexofdata, ...] = read_image(path1train[i], path2train[i])
            with tf.GradientTape(persistent=True) as tape:
                out = SEGPAVER(imageinput, training=True)
                loss = BCE(maskout, out)

            gradients = tape.gradient(loss, SEGPAVER.trainable_variables)
            optimizer.apply_gradients(zip(gradients, SEGPAVER.trainable_variables))

            indexofdata += batch_size
            if indexofdata + batch_size >= len(path1train):
                break
        tf.print(loss, '\t', epochs)
        epochs -= 1    
        if epochs % 1 == 0:
            gene_images(SEGPAVER, imageinput, epochs)
        if epochs == 10:
            optimizer = tf.keras.optimizers.Adam(lr * 0.1, beta_1=0.5)
        if epochs == 15:
            optimizer = tf.keras.optimizers.Adam(lr * 0.01, beta_1=0.5)
        if epochs == 25:
            optimizer = tf.keras.optimizers.Adam(lr * 0.001, beta_1=0.5)
        if min_loss > loss:
            min_loss = loss
            SEGPAVER.save('+_maskenhance_ken_size%d_each_layers%d_dilation_rate%d.h5'%(ken_size[0], each_layers, dilation_rate))
            SEGPAVER.save_weights('+_maskenhance_ken_size%d_each_layers%d_dilation_rate%dweight.h5'%(ken_size[0], each_layers, dilation_rate))
        validimg = np.empty((50, inputsize[0], inputsize[1], 1))
        validmask = np.empty((50, masksize[0], masksize[1], 1))
        for i in range(0, 50):
            validimg[i, ...], \
                    validmask[i, ...] = read_image(path1val[i], path2val[i])
        pred = SEGPAVER(validimg)
        
        print(pred.shape)
        print(tf.experimental.numpy.sum(tf.keras.metrics.binary_accuracy(validmask, pred, threshold=0.5))    / pred.shape[0] / pred.shape[1] /pred.shape[2],end='\t') # / pred.shape[0] * pred.shape[1] *pred.shape[2] 
        print(tf.keras.metrics.Precision()(validmask, pred), end='\t') 
        print(tf.keras.metrics.Recall()(validmask, pred))           
                
        