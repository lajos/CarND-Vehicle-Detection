import utils, histogram, binning, hog, feat, classify, heatmap
import constants as c
import glob, pickle, time, sys
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    # vehicle_images = utils.unpickle_data(c.vehicles_train_data_p)
    # non_vehicle_images = utils.unpickle_data(c.non_vehicles_train_data_p)

    vehicle_images = make_image_pickle(c.vehicles_train_data_folder, c.vehicles_train_data_p, (64,64,3))
    non_vehicle_images = make_image_pickle(c.non_vehicles_train_data_folder, c.non_vehicles_train_data_p, (64,64,3))
    vehicle_hists = histogram.multispace_histograms_images(vehicle_images, c.vehicles_histograms_p)
    non_vehicle_hists = histogram.multispace_histograms_images(non_vehicle_images, c.non_vehicles_histograms_p)
    vehicle_sbins = binning.multispace_spatial_bin_images(vehicle_images, c.vehicles_spatial_bins_p)
    non_vehicle_sbins = binning.multispace_spatial_bin_images(non_vehicle_images, c.non_vehicles_spatial_bins_p)
    vehicle_hogs = hog.multispace_hog_images(vehicle_images, c.vehicles_hog_p)
    non_vehicle_hogs = hog.multispace_hog_images(non_vehicle_images, c.non_vehicles_hog_p)


def process_image(img_bgr, svc, X_scaler, use_features, scan_params):
    detections = None
    for sp in scan_params:
        d = classify.find_vehicles(img_bgr, svc, X_scaler, use_features, sp[0], sp[1], sp[2])
        if d is not None:
            if detections is None:
                detections = d
            else:
                detections = np.concatenate((detections, d), axis=0)

    if detections is None:
        return img_bgr, np.zeros_like(img_bgr), None

    hmap = heatmap.HeatMap(img_bgr.shape[:2])

    print(detections.shape)

    hmap.add(detections)
    hmap.threshold(10)
    hmap.label()

    clipped = hmap.get_clipped()

    draw_img = hmap.draw_labeled_bboxes(img_bgr.copy())

    return draw_img, clipped, detections


def test_image(svc, X_scaler, use_features, scan_params):
    img_bgr = cv2.imread('{}/0001.png'.format(c.test_video_images_folder))

    draw_img, clipped, detections = process_image(img_bgr, svc, X_scaler, use_features, scan_params)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(clipped, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()
    cv2.waitKey()

def process_video(video_filename, svc, X_scaler, use_features, scan_params):

    def process_video_image(img):
        global current_video_frame
        img = utils.img_reverse_channels(img)
        draw_img, clipped, detections = process_image(img, svc, X_scaler, use_features, scan_params)
        cv2.imwrite('output_images/debug/{:04d}.png'.format(current_video_frame), clipped)
        current_video_frame += 1
        return utils.img_reverse_channels(draw_img)

    from moviepy.editor import VideoFileClip
    global current_video_frame, poly_log

    current_video_frame = 0
    utils.make_dir(c.output_folder)

    input_video = video_filename
    output_video = '{}/{}'.format(c.output_folder, video_filename)

    clip = VideoFileClip(input_video)
    processed_clip = clip.fl_image(process_video_image)
    processed_clip.write_videofile(output_video, audio=False)

    # write polygon data for debugging
    # utils.write_csv('poly_log.csv', poly_log)

if __name__=='__main__':
    do_preprocess = False
    do_fit = False

    if do_preprocess:
        preprocess()

    # use_features = {
    #     c.hists: [[c.hls_index, 1],
    #               [c.hls_index, 2]],
    #     c.sbins: [c.hls_index,
    #               c.xyz_index,
    #               c.luv_index],
    #     c.hogs: [[c.luv_index, 0],
    #             [c.luv_index, 1],
    #             [c.luv_index, 2]]
    # }

    use_features = {
        c.hists: [[c.luv_index, 0],
                  [c.luv_index, 1],
                  [c.luv_index, 2]],
        c.sbins: [c.luv_index],
        c.hogs: [[c.luv_index, 0],
                [c.luv_index, 1],
                [c.luv_index, 2]]
    }


    X_scaler = None
    svc = None

    if do_fit:
        X, y, X_scaler = feat.get_features(use_features)

        from sklearn.svm import LinearSVC
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        import time

        print('X:', X.shape)
        print('y:', y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        svc = LinearSVC()
        # svc = SVC(probability=True)
        t=time.time()
        svc.fit(X_train, y_train)

        print('SVC fit time: {:.2f}s'.format(time.time()-t))
        print('SVC test accuracy: {:.4f}'.format(svc.score(X_test, y_test)))

        utils.pickle_data(c.x_scaler_p, X_scaler)
        utils.pickle_data(c.svm_p, svc)
    else:
        X_scaler = utils.unpickle_data(c.x_scaler_p)
        svc = utils.unpickle_data(c.svm_p)

    # define scan regions and window sizes (y_from, y_to, window_size)
    # scan_params = [
    #     (400,400+64,64),
    #     (400,400+128,96),
    #     (400,400+192,128),
    #     (400,400+256,160),
    #     (400,400+256,256)
    # ]
    scan_params = [
        (400,400+128,48),
        (400,400+256,64),
        (400,400+256,128),
        (400,400+256,160),
        (400,400+256,256)
    ]

    # test_image(svc, X_scaler, use_features, scan_params)

    process_video(c.test_video, svc, X_scaler, use_features, scan_params)
    # process_video(c.project_video, svc, X_scaler, use_features, scan_params)

