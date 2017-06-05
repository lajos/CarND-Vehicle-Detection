import cv2
import constants as c
import matplotlib.pyplot as plt
import utils
import numpy as np


def spatial_bin(img, bin_size=(32,32)):
    features = cv2.resize(img, bin_size)
    return features.ravel()

def multispace_spatial_bin(img_bgr, bin_size=(32,32)):
    """calculate spatial bins for RGB, HLS, xyz and Luv color spaces"""
    m_features = []

    img = img_bgr
    m_features.append(spatial_bin(img, bin_size))

    img = utils.img_bgr2hls(img_bgr)
    m_features.append(spatial_bin(img, bin_size))

    img = utils.img_bgr2xyz(img_bgr)
    m_features.append(spatial_bin(img, bin_size))

    img = utils.img_bgr2luv(img_bgr)
    m_features.append(spatial_bin(img, bin_size))

    return(np.array(m_features))

def multispace_spatial_bin_images(images, output_file=None, bin_size=(32,32)):
    """calculate multi colorspace spatial bins for a set of images and optionally pickle"""
    if output_file is not None:
        print('calculate multispace spatial bins, pickle to:', output_file)

    mf_hists = []

    utils.print_progress_bar (0, len(images), prefix = 'multispace spatial bins:')

    for i in range(len(images)):
        img = images[i]
        mf_hists.append(multispace_spatial_bin(img, bin_size))
        if i%100==0:
            utils.print_progress_bar (i, len(images), prefix = 'multispace spatial bins:')

    mf_hists = np.array(mf_hists)

    if output_file is not None:
        utils.pickle_data(output_file, mf_hists)

    utils.print_progress_bar (len(images), len(images), prefix = 'multispace spatial bins:')

    return mf_hists

def plot_bin(features):
    plt.plot(features)
    plt.show()

if __name__=='__main__':
    img_bgr = cv2.imread('{}/GTI_MiddleClose/image0185.png'.format(c.vehicles_train_data_folder))

    cv2.imshow('binning', img_bgr)
    cv2.waitKey()

    features = spatial_bin(img_bgr)

    plot_bin(features)

    m_features = multispace_spatial_bin(img_bgr)

    print(m_features)
    print(m_features.shape)

