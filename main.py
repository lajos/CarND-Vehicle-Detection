import utils, histogram, binning, hog, feat, classify, heatmap
import constants as c
import glob, pickle, time, sys, os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# suppress scientific notation
np.set_printoptions(suppress=True)

def make_image_pickle(folder, output_file, expected_shape=None):
    """save images into a pickle file for faster loading"""
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
    """preprocess training images, create all histograms, sbins and hogs"""
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
    """detect vehicle areas in an image"""
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

    hmap.add(detections)
    hmap.threshold(0)
    hmap.label()

    clipped = hmap.get_clipped()

    draw_img = hmap.draw_labeled_bboxes(img_bgr.copy())

    return draw_img, clipped, detections


def test_image(svc, X_scaler, use_features, scan_params):
    """run detection on a single image for testing"""
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
    """detect vehicles in a video file"""
    all_detections = []

    def process_video_image(img):
        global current_video_frame
        img = utils.img_reverse_channels(img)
        draw_img, clipped, detections = process_image(img, svc, X_scaler, use_features, scan_params)
        all_detections.append(detections)
        cv2.imwrite('output_images/debug/img{:04d}.png'.format(current_video_frame), draw_img)
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

    # save detections for debugging
    utils.pickle_data(c.all_detections_p, all_detections)


def process_video_folder(video_folder, svc, X_scaler, use_features, scan_params, start_frame=0, end_frame=None):
    """detect vehicle areas in a folder of images"""
    all_detections = utils.unpickle_data(c.all_detections_p)

    video_glob = glob.glob('{}/*.png'.format(video_folder))
    video_glob.sort()
    if end_frame is None:
        end_frame = len(video_glob)

    start_time = time.time()

    for current_frame in range(start_frame, end_frame):
        video_image_name = video_glob[current_frame]
        video_image_basename = os.path.basename(video_image_name)

        elapsed = (time.time() - start_time)/60
        eta = 0
        if current_frame>0:
            eta = (elapsed/current_frame)*(len(video_glob)-current_frame)

        print('processing: {} - elapsed: {:.2f}min - eta: {:.2f}min'.format(video_image_basename,elapsed,eta))

        img_bgr = cv2.imread(video_image_name)
        draw_img, clipped, detections = process_image(img_bgr, svc, X_scaler, use_features, scan_params)

        if len(all_detections)<=current_frame:
            all_detections.append(detections)
        else:
            all_detections[current_frame]=detections

        cv2.imwrite('output_images/debug/img{}'.format(video_image_basename), draw_img)
        cv2.imwrite('output_images/debug/hmp{}'.format(video_image_basename), clipped)

        current_frame += 1

    # save detections for debugging
    utils.pickle_data(c.all_detections_p, all_detections)

def process_detections(start_frame=0, end_frame=None):
    """use heatmap to find vehicles based on detections, average detection over frame range"""
    video_folder = c.project_video_images_folder
    video_glob = glob.glob('{}/*.png'.format(video_folder))
    video_glob.sort()

    if end_frame is None:
        end_frame = len(video_glob)

    all_detections = utils.unpickle_data(c.all_detections_p)

    utils.print_progress_bar (0, len(video_glob), prefix = 'process detections:')

    for i in range(start_frame, end_frame):
        video_image_name = video_glob[i]
        video_image_basename = os.path.basename(video_image_name)

        img_bgr = cv2.imread(video_image_name)
        img = img_bgr.copy()

        hm = heatmap.HeatMap(img_bgr.shape[:2])
        for o in range(max(0,i-16),i):
            hm.add(all_detections[o])
        hm.threshold(20)
        hm.label()

        img_bgr = img_bgr // 1.5
        img_bgr = hm.draw_labeled_bboxes(img_bgr, min_w=32, min_h=32)
        img = hm.draw_labeled_bboxes(img, min_w=65, min_h=45)

        if all_detections[i] is not None:
            for o in all_detections[i]:
                cv2.rectangle(img_bgr,tuple(o[0]),tuple(o[1]),(255,0,0),1)

        cv2.imwrite('output_images/debug/dbg{}'.format(video_image_basename), img_bgr)
        cv2.imwrite('output_images/debug/out{}'.format(video_image_basename), img)

        utils.print_progress_bar (i, len(video_glob), prefix = 'process detections:', suffix='frame: {:d}'.format(i))


    utils.print_progress_bar (len(video_glob), len(video_glob), prefix = 'process detections:')

def main():
    do_preprocess = True
    do_fit = True
    do_process_video_folder = True
    do_process_detections = True
    start_frame=0
    end_frame=None

    if do_preprocess:
        preprocess()

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        svc = LinearSVC(verbose=1, max_iter=5000, random_state=42)
        # svc = LinearSVC(penalty='l1', dual=False, verbose=1, random_state=42)
        # svc = LinearSVC(dual=False, verbose=1, random_state=42)
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

    # test_image(svc, X_scaler, use_features, scan_params)

    # process_video(c.project_video, svc, X_scaler, use_features, scan_params)

    if do_process_video_folder:
        process_video_folder(c.project_video_images_folder, svc, X_scaler, use_features, scan_params, start_frame=start_frame, end_frame=end_frame)

    if do_process_detections:
        process_detections(start_frame=start_frame, end_frame=end_frame)



if __name__=='__main__':
    main()
