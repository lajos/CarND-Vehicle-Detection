import cv2
from skimage.feature import hog
import constants as c
import matplotlib.pyplot as plt
import utils
import numpy as np

def hog_features(img, orient=8, pix_per_cell=8, cell_per_block=2, transform_sqrt=True, vis=False, flatten=False):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt,
                                  visualise=True, block_norm='L2-Hys')
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt,
                       visualise=False, block_norm='L2-Hys')
        if flatten:
            features = np.ravel(features)
        return features

def hog_channels(img, orient=8, pix_per_cell=8, cell_per_block=2, transform_sqrt=True, flatten=False):
    c_features = []
    for i in range(img.shape[2]):
        c_features.append(hog_features(img[:,:,i], orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       transform_sqrt=transform_sqrt,
                                       flatten=flatten,
                                       vis=False))
    return np.array(c_features)

def multispace_hog_features(img_bgr, orient=8, pix_per_cell=8, cell_per_block=2, transform_sqrt=True, flatten=False):
    """calculate hogs for RGB, HLS, xyz and Luv color spaces"""
    m_features = []
    img = img_bgr
    m_features.append(hog_channels(img, orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   transform_sqrt=transform_sqrt,
                                   flatten=flatten))

    img = utils.img_bgr2hls(img_bgr)
    m_features.append(hog_channels(img, orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   transform_sqrt=transform_sqrt,
                                   flatten=flatten))

    img = utils.img_bgr2xyz(img_bgr)
    m_features.append(hog_channels(img, orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   transform_sqrt=transform_sqrt,
                                   flatten=flatten))

    img = utils.img_bgr2luv(img_bgr)
    m_features.append(hog_channels(img, orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   transform_sqrt=transform_sqrt,
                                   flatten=flatten))

    return(np.array(m_features))

def multispace_hog_images(images, output_file=None, orient=8, pix_per_cell=8, cell_per_block=2, transform_sqrt=True, flatten=True):
    """calculate hog for a set of images and optionally pickle"""
    if output_file is not None:
        print('calculate hog images, pickle to:', output_file)

    hogs = []

    utils.print_progress_bar (0, len(images), prefix = 'hog:')

    for i in range(len(images)):
        img_bgr = images[i]
        hogs.append(multispace_hog_features(img_bgr, orient=orient,
                                            pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            transform_sqrt=transform_sqrt,
                                            flatten=flatten))
        if i%100==0:
            utils.print_progress_bar (i, len(images), prefix = 'hog:')

    hogs = np.array(hogs)

    if output_file is not None:
        utils.pickle_data(output_file, hogs)

    utils.print_progress_bar (len(images), len(images), prefix = 'hog:')

    return hogs

if __name__ == '__main__':
    img_bgr = cv2.imread('{}/GTI_MiddleClose/image0185.png'.format(c.vehicles_train_data_folder))

    # cv2.imshow('binning', img_bgr)
    # cv2.waitKey()

    img_l = utils.img_bgr2l(img_bgr)

    features, hog_image = hog_features(img_l, orient=8,
                            pix_per_cell=8, cell_per_block=2,
                            vis=True, transform_sqrt=True)

    print(features.shape)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img_l, cmap='gray')
    plt.title('source')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('hog')
    plt.show()