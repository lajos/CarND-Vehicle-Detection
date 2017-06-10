import utils, hog, histogram, binning
import constants as c
import numpy as np
import cv2, sys, time, os, glob, time
import matplotlib.pyplot as plt
import heatmap

def slide_windows(images, hogs, use_features, scale):
    """slide window over image and find vehicle areas"""
    img_shape = np.array([x for x in images if x is not None][0].shape)
    img_w = img_shape[1]
    img_h = img_shape[0]

    # print('img shape:',img_shape)

    # calculate hog block counts
    n_blocks = (img_shape[:2] / c.hog_pix_per_cell).astype(int) - c.hog_cell_per_block + 1
    n_x_blocks = n_blocks[1]
    n_y_blocks = n_blocks[0]

    # print('n_blocks:', n_blocks)

    window_size = c.sample_size
    n_blocks_per_window = (window_size // c.hog_pix_per_cell) - c.hog_cell_per_block + 1
    cells_per_step = 4  # Instead of overlap, define how many cells to step

    n_steps = ((n_blocks - n_blocks_per_window) / cells_per_step).astype(int)
    n_x_steps = n_steps[1]
    n_y_steps = n_steps[0]

    min_lightness = 14 * window_size * window_size

    # print('n_blocks_per_window:',n_blocks_per_window)
    # print('n_steps:', n_steps)

    # print('sliding:', window_size*scale, n_blocks, n_blocks_per_window, n_steps)

    detections = []
    for x_pos in range(n_x_blocks%cells_per_step, n_x_blocks-n_blocks_per_window+1, cells_per_step):
        for y_pos in range(n_y_blocks%cells_per_step, n_y_blocks-n_blocks_per_window+1, cells_per_step):

            # print('xy pos:',x_pos,y_pos)

            x_left = x_pos * c.hog_pix_per_cell
            y_top = y_pos * c.hog_pix_per_cell

            # reject very dark samples
            sub_img = images[c.luv_index][:,:,0][y_top:y_top+window_size, x_left:x_left+window_size]
            if np.sum(sub_img) < min_lightness:
                continue

            features=np.zeros(1, dtype=np.float32)       # "empty" features to concat to

            for cspace, channel in use_features[c.hists]:
                sub_img = images[cspace][y_top:y_top+window_size, x_left:x_left+window_size]
                f = histogram.color_histograms(sub_img)[channel]
                features = np.concatenate((features, f))

            for cspace in use_features[c.sbins]:
                sub_img = images[cspace][y_top:y_top+window_size, x_left:x_left+window_size]
                f = binning.spatial_bin(sub_img)
                features = np.concatenate((features, f))

            for h in hogs:
                hf = h[y_pos:y_pos+n_blocks_per_window, x_pos:x_pos+n_blocks_per_window].ravel()
                features = np.concatenate((features, hf))

            features = features[1:]  # remove 0 column that was created for placeholder

            features = features.reshape(1,-1)

            features_time = time.time()

            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(x_left*scale)
                ytop_draw = np.int(y_top*scale)
                win_draw = np.int(window_size*scale)
                detections.append([(xbox_left, ytop_draw),(xbox_left+win_draw,ytop_draw+win_draw)])
                # sub_img = images[c.luv_index][:,:,0][y_top:y_top+window_size, x_left:x_left+window_size]
                # print(np.sum(sub_img))

    return detections


def get_channels(img_bgr, use_features):
    """prepare all image spaces and channels"""
    images = []
    channels = []

    all_spaces = {}

    for i in use_features[c.hists]:
        all_spaces[i[0]]=True

    for i in use_features[c.sbins]:
        all_spaces[i]=True

    for i in use_features[c.hogs]:
        all_spaces[i[0]]=True

    if c.bgr_index in all_spaces.keys():
        img = img_bgr
        images.append(img)
        channels.append(cv2.split(img))
    else:
        images.append(None)
        channels.append(None)

    if c.hls_index in all_spaces.keys():
        img = utils.img_bgr2hls(img_bgr)
        images.append(img)
        channels.append(cv2.split(img))
    else:
        images.append(None)
        channels.append(None)

    if c.xyz_index in all_spaces.keys():
        img = utils.img_bgr2xyz(img_bgr)
        images.append(img)
        channels.append(cv2.split(img))
    else:
        images.append(None)
        channels.append(None)

    if c.luv_index in all_spaces.keys():
        img = utils.img_bgr2luv(img_bgr)
        images.append(img)
        channels.append(cv2.split(img))
    else:
        images.append(None)
        channels.append(None)

    return images, channels

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


def find_vehicles(img_bgr, svc_, X_scaler_, use_features, y_from=400, y_to=656, window_size=64):
    """detect vehicles in an image, scaling image to allow smaller/larger detection window"""
    global X_scaler, svc
    X_scaler = X_scaler_
    svc = svc_

    scale = window_size / c.sample_size
    # print(scale)

    img_region = img_bgr[y_from:y_to,:,:]
    if window_size!=c.sample_size:
        # img_region = cv2.resize(img_region, (np.int(img_region.shape[1]/scale), np.int(img_region.shape[0]/scale)), interpolation=cv2.INTER_LINEAR)
        if scale > 1:
            img_region = cv2.resize(img_region, (np.int(img_region.shape[1]/scale), np.int(img_region.shape[0]/scale)), interpolation=cv2.INTER_AREA)
        else:
            img_region = cv2.resize(img_region, (np.int(img_region.shape[1]/scale), np.int(img_region.shape[0]/scale)), interpolation=cv2.INTER_LINEAR)

    # cv2.imshow('test', img_region)
    # cv2.waitKey()

    images, channels = get_channels(img_region, use_features)
    hogs = get_hogs(channels, use_features)

    detections = slide_windows(images, hogs, use_features, scale)

    # print('number of detections:',len(detections))

    if len(detections)==0:
        return None

    detections = np.array(detections)
    detections[:,:,1] += y_from

    return detections


def test_run(img_bgr, show_result=True, save_false=False, false_threshold=5):
    """test detection on an image"""
    # img_bgr = cv2.imread('{}/0250.png'.format(c.project_video_images_folder))
    img_ori = img_bgr.copy()

    # cv2.imshow('test', img_bgr)
    # cv2.waitKey()

    X_scaler_ = utils.unpickle_data(c.x_scaler_p)
    svc_ = utils.unpickle_data(c.svm_p)

    use_features = {
        c.hists: [[c.bgr_index, 0],
                  [c.luv_index, 0],
                  [c.luv_index, 1],
                  [c.luv_index, 2]],
        c.sbins: [ c.luv_index,
                   c.bgr_index],
        c.hogs:  [[c.luv_index, 0],
                  [c.luv_index, 1],
                  [c.luv_index, 2]]
    }

    scan_params = [
        (400,400+128,48),
        (400,400+256,64),
        (400,400+256,80),
        (400,400+256,96),
        (400,400+256,112),
        (400,400+256,128),
        (400,400+256,144),
        (400,400+256,160)
    ]

    start_time = time.time()

    detections = None
    for sp in scan_params:
        d = find_vehicles(img_bgr, svc_, X_scaler_, use_features, sp[0], sp[1], sp[2])
        if d is not None:
            if detections is None:
                detections = d
            else:
                detections = np.concatenate((detections, d), axis=0)
    print('detection time: {:.2f}'.format(time.time()-start_time))

    print(detections.shape)

    if detections is None:
        print('no vehicles detected')
        return

    utils.pickle_data(c.test_detections_p, detections)

    for i in detections:
        cv2.rectangle(img_bgr,tuple(i[0]),tuple(i[1]),(0,0,255),1)

    hmap = heatmap.HeatMap(img_bgr.shape[:2])
    hmap.add(detections)

    h = hmap.heatmap.copy().astype(np.uint8)

    for d in detections:
        x_pos = d[0][0]
        y_pos = d[0][1]
        x_width = d[1][0] - x_pos
        y_width = d[1][1] - y_pos

        if np.max(h[y_pos:y_pos+y_width, x_pos:x_pos+x_width]) < false_threshold:
            print('false detection:',d[0])
            img_false = img_ori[y_pos:y_pos+y_width, x_pos:x_pos+x_width].copy()
            img_false = cv2.resize(img_false, (64,64))
            false_glob = glob.glob('{}/*.png'.format(c.non_vehicles_auto_folder))
            false_name = '0000.png'
            if len(false_glob):
                false_glob.sort()
                false_name = int(os.path.basename(false_glob[-1]).split('.')[0])+1
                false_name = '{:04d}.png'.format(false_name)
            if save_false:
                print(false_name)
                cv2.imwrite('{}/{}'.format(c.non_vehicles_auto_folder, false_name), img_false)


    # cv2.imshow('heat',h*16)

    if show_result:
        cv2.imshow('test', img_bgr)
        cv2.waitKey()



if __name__=='__main__':
    img_bgr = cv2.imread('{}/0502.png'.format(c.project_video_images_folder))
    test_run(img_bgr, show_result=True, save_false=False)
    cv2.imwrite('test.png',img_bgr)

    # for i in range(531,549):
    #     img_bgr = cv2.imread('{}/{:04d}.png'.format(c.project_video_images_folder, i))
    #     test_run(img_bgr, show_result=False, save_false=True, false_threshold=21)

