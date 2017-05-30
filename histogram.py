import cv2
import numpy as np
import constants as c
import utils
import matplotlib.pyplot as plt

def color_histograms(img, nbins=32):
    """calculate histogram for all channels, returns list of histograms"""
    features = []
    for i in range(img.shape[2]):
        hist = np.histogram(img[:,:,i], bins=nbins, range=(0, 256))
        features.append(hist[0])
    return features

def multispace_histograms(img_bgr, nbins=32):
    """calculate histograms for RGB, HLS, xyz and Luv color spaces"""
    m_features = []

    img = img_bgr
    m_features.append(color_histograms(img, nbins))

    img = utils.img_bgr2hls(img_bgr)
    m_features.append(color_histograms(img, nbins))

    img = utils.img_bgr2xyz(img_bgr)
    m_features.append(color_histograms(img, nbins))

    img = utils.img_bgr2luv(img_bgr)
    m_features.append(color_histograms(img, nbins))

    return(np.array(m_features))

def multispace_histograms_images(images, output_file=None, nbins=32):
    """calculate multi colorspace histograms for a set of images and optionally pickle"""
    if output_file is not None:
        print('calculate multispace histograms, pickle to:', output_file)

    mf_hists = []

    utils.print_progress_bar (0, len(images), prefix = 'multispace histogram:')

    for i in range(len(images)):
        img = images[i]
        mf_hists.append(multispace_histograms(img, nbins))
        if i%100==0:
            utils.print_progress_bar (i, len(images), prefix = 'multispace histogram:')

    mf_hists = np.array(mf_hists)

    if output_file is not None:
        utils.pickle_data(output_file, mf_hists)

    utils.print_progress_bar (len(images), len(images), prefix = 'multispace histogram:')

    return mf_hists

def plot_histograms(features, hist_names=('hist0','hist1','hist2')):
    fig = plt.figure(figsize=(12,3))
    subplot_id=101+len(features)*10
    for i in range(len(features)):
        plt.subplot(subplot_id+i)
        plt.title(hist_names[i])
        plt.bar(np.arange(len(features[i])), features[i])
    plt.show()

if __name__=='__main__':
    img_bgr = cv2.imread('{}/GTI_MiddleClose/image0185.png'.format(c.vehicles_train_data_folder))

    # cv2.imshow('hist',img_bgr)
    # cv2.waitKey()

    # img = img_bgr
    # features = color_histograms(img)
    # plot_histograms(features,('B','G','R'))

    # img = utils.img_bgr2hls(img_bgr)
    # features = color_histograms(img)
    # plot_histograms(features,('H','L','S'))

    # img = utils.img_bgr2xyz(img_bgr)
    # features = color_histograms(img)
    # plot_histograms(features,('x','y','z'))

    # img = utils.img_bgr2luv(img_bgr)
    # features = color_histograms(img)
    # plot_histograms(features,('L','u','v'))

    mf = multispace_histograms(img_bgr)

    print(mf)
    print(mf.shape)

