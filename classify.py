import utils, hog
import constants as c
import numpy as np
import cv2, sys
import matplotlib.pyplot as plt


def find_cars_ori(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return draw_img


X_scaler = None
svc = None

def slide_windows(channels, hogs, use_features):
    img_shape = np.array(channels[c.bgr_index][0].shape)
    img_w = img_shape[1]
    img_h = img_shape[0]

    # calculate hog box counts
    n_blocks = (img_shape / c.hog_pix_per_cell).astype(int) - c.hog_cell_per_block + 1
    n_x_blocks = n_blocks[1]
    n_y_blocks = n_blocks[0]

    # hog features per block
    n_feat_per_block = c.hog_orient*c.hog_cell_per_block**2


# we are here

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return draw_img


def get_channels(img_bgr):
    """prepare all image channels"""
    channels = []

    img = img_bgr
    channels.append(cv2.split(img))

    img = utils.img_bgr2hls(img_bgr)
    channels.append(cv2.split(img))

    img = utils.img_bgr2xyz(img_bgr)
    channels.append(cv2.split(img))

    img = utils.img_bgr2luv(img_bgr)
    channels.append(cv2.split(img))

    # print('channels:',len(channels))
    # print('bgr channels:',len(channels[c.bgr_index]), channels[c.bgr_index][0].shape)
    # print('hls channels:',len(channels[c.hls_index]), channels[c.hls_index][0].shape)
    # print('xyz channels:',len(channels[c.xyz_index]), channels[c.xyz_index][0].shape)
    # print('luv channels:',len(channels[c.luv_index]), channels[c.luv_index][0].shape)

    return channels

def get_hogs(channels, use_features):
    """prepare used hogs"""
    hogs = []
    for c_space, ch_index in use_features[c.hogs]:
        # print('prepare hog:',c_space, ch_index)
        h = hog.hog_features(channels[c_space][ch_index],
                                      orient=c.hog_orient,
                                      pix_per_cell=c.hog_pix_per_cell,
                                      cell_per_block=c.hog_cell_per_block,
                                      transform_sqrt=c.hog_transform_sqrt,
                                      vis=False, flatten=False)
        hogs.append(h)
    return hogs


def find_vehicles(img_bgr, svc_, X_scaler_, use_features):
    global X_scaler, svc
    X_scaler = X_scaler_
    svc = svc_

    y_from = 400
    y_to = 656
    scale = 0.5

    img_region = img_bgr[y_from:y_to,:,:]
    if scale != 1:
        img_region = cv2.resize(img_region, (np.int(img_region.shape[1]*scale), np.int(img_region.shape[0]*scale)))

    # cv2.imshow('test', img_region)
    # cv2.waitKey()

    channels = get_channels(img_region)
    hogs = get_hogs(channels, use_features)

    slide_windows(channels, hog, use_features)



if __name__=='__main__':
    img_bgr = cv2.imread('{}/0001.png'.format(c.test_video_images_folder))
    # cv2.imshow('test', img_bgr)
    # cv2.waitKey()

    X_scaler_ = utils.pickle_data(c.x_scaler_p, X_scaler)
    svc_ = utils.pickle_data(c.svm_p, svc)

    use_features = {
        c.hists: [[c.hls_index, 1],
                  [c.hls_index, 2]],
        c.sbins: [c.hls_index,
                  c.xyz_index,
                  c.luv_index],
        c.hogs: [[c.luv_index, 0],
                [c.luv_index, 1],
                [c.luv_index, 2]]
    }

    find_vehicles(img_bgr, svc_, X_scaler_, use_features)

    # out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # plt.imshow(out_img)