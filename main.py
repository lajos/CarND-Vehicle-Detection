import utils, histogram, binning, hog
import constants as c
import glob, pickle, time, sys
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# suppress scientific notation
np.set_printoptions(suppress=True)

def make_image_pickle(folder, output_file, expected_shape=None):
    images = []
    print('make image pickle: {}/** -> {}'.format(folder, output_file))
    for i in glob.iglob('{}/**/*.png'.format(folder), recursive=True):
        img = cv2.imread(i)
        if expected_shape is not None:
            if not img.shape==expected_shape:
                utils.warning()
        images.append(img)
    print('   got {} images'.format(len(images)))
    images = np.array(images)
    utils.pickle_data(output_file, np.array(images))
    return images

def preprocess():
    # preload image data to speed up processing for testing
    vehicle_images = utils.unpickle_data(c.vehicles_train_data_p)
    non_vehicle_images = utils.unpickle_data(c.non_vehicles_train_data_p)

    # vehicle_images = make_image_pickle(c.vehicles_train_data_folder, c.vehicles_train_data_p, (64,64,3))
    # non_vehicle_images = make_image_pickle(c.non_vehicles_train_data_folder, c.non_vehicles_train_data_p, (64,64,3))
    # vehicle_hists = histogram.multispace_histograms_images(vehicle_images, c.vehicles_histograms_p)
    # non_vehicle_hists = histogram.multispace_histograms_images(non_vehicle_images, c.non_vehicles_histograms_p)
    # vehicle_sbins = binning.multispace_spatial_bin_images(vehicle_images, c.vehicles_spatial_bins_p)
    # non_vehicle_sbins = binning.multispace_spatial_bin_images(non_vehicle_images, c.non_vehicles_spatial_bins_p)
    vehicle_hogs = hog.multispace_hog_images(vehicle_images, c.vehicles_hog_p)
    non_vehicle_hogs = hog.multispace_hog_images(non_vehicle_images, c.non_vehicles_hog_p)


def combine_features(hists, sbins, hogs, sel_hists=[], sel_sbins=[], sel_hogs=False):
    features=np.zeros((hists.shape[0],1), dtype=np.float32)       # "empty" features to concat to

    # for cspace, channel in sel_hists:
    #     features = np.concatenate((features, utils.scale_mean_std(hists[:, cspace, channel])),axis=1)

    # for cspace in sel_sbins:
    #     features = np.concatenate((features, utils.scale_mean_std(sbins[:, cspace])),axis=1)

    # if sel_hogs:
    #     features = np.concatenate((features, utils.scale_mean_std(hogs)),axis=1)

    for cspace, channel in sel_hists:
        features = np.concatenate((features, hists[:, cspace, channel]),axis=1)

    for cspace in sel_sbins:
        features = np.concatenate((features, sbins[:, cspace]),axis=1)

    if sel_hogs:
        features = np.concatenate((features, hogs),axis=1)


    features = features[:,1:]  # remove 0 column that was created for placeholder

    return features


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

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

if __name__=='__main__':
    preprocess()
    sys.exit(0)

    vehicle_images = utils.unpickle_data(c.vehicles_train_data_p)
    non_vehicle_images = utils.unpickle_data(c.non_vehicles_train_data_p)
    print('vehicle images: ',vehicle_images.shape)
    print('non_vehicle images: ',non_vehicle_images.shape)

    vehicle_hists = utils.unpickle_data(c.vehicles_histograms_p)
    non_vehicle_hists = utils.unpickle_data(c.non_vehicles_histograms_p)
    print('vehicle hists: ',vehicle_hists.shape)
    print('non_vehicle hists: ',non_vehicle_hists.shape)

    vehicle_sbins = utils.unpickle_data(c.vehicles_spatial_bins_p)
    non_vehicle_sbins = utils.unpickle_data(c.non_vehicles_spatial_bins_p)
    print('vehicle bins: ',vehicle_sbins.shape)
    print('non_vehicle bins: ',non_vehicle_sbins.shape)

    vehicle_hogs = utils.unpickle_data(c.vehicles_hog_p)
    non_vehicle_hogs = utils.unpickle_data(c.non_vehicles_hog_p)
    print('vehicle hog: ',vehicle_hogs.shape)
    print('non_vehicle hog: ',non_vehicle_hogs.shape)

    vehicle_features = combine_features(vehicle_hists, vehicle_sbins, vehicle_hogs,
                                        sel_hists = [[c.hls_index, 2],
                                                    [c.xyz_index, 0],
                                                    [c.luv_index, 0]],
                                        sel_sbins = [c.hls_index,
                                                     c.xyz_index,
                                                     c.luv_index],
                                        sel_hogs=True)

    non_vehicle_features = combine_features(non_vehicle_hists, non_vehicle_sbins, non_vehicle_hogs,
                                            sel_hists = [[c.hls_index, 2],
                                                        [c.xyz_index, 0],
                                                        [c.luv_index, 0]],
                                            sel_sbins = [c.hls_index,
                                                        c.xyz_index,
                                                        c.luv_index],
                                            sel_hogs=True)


    print('vehicle feautures:',vehicle_features.shape)
    print('non_vehicle feautures:',non_vehicle_features.shape)

    X = np.concatenate((vehicle_features, non_vehicle_features))
    y = np.concatenate((np.ones(vehicle_features.shape[0]), np.zeros(non_vehicle_features.shape[0])))

    X_scaler = StandardScaler().fit(X)

    X = X_scaler.transform(X)


    print('X:', X.shape)
    print('y:', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # t=time.time()
    # n_predict = 1000
    # print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    # print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    # t2 = time.time()
    # print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
