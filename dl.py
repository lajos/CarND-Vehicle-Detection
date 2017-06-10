from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten
import utils
from heatmap import HeatMap
import constants as c
import numpy as np
from sklearn.utils import shuffle
import sys, cv2, time, glob, os
import matplotlib.pylab as plt

# suppress scientific notation
np.set_printoptions(suppress=True)


def img_normalize(img):
    """normalize image to [-0.5...0.5] range"""
    return img/255.0-0.5

def preprocess_images(images):
    utils.print_progress_bar (0, len(images), prefix = 'preprocess images:')
    preprocessed = []
    for i in range(len(images)):
        img = images[i]
        img = utils.img_bgr2luv(img)
        img = img_normalize(img)
        preprocessed.append(img)
        utils.print_progress_bar (i, len(images), prefix = 'preprocess images:')
    utils.print_progress_bar (len(images), len(images), prefix = 'preprocess images:')
    return np.array(preprocessed)


def load_train_data(preprocess=True):
    vehicles=None
    non_vehicles=None
    if preprocess:
        vehicles = utils.unpickle_data(c.vehicles_train_data_p)
        non_vehicles = utils.unpickle_data(c.non_vehicles_train_data_p)
        vehicles = preprocess_images(vehicles)
        non_vehicles = preprocess_images(non_vehicles)
        utils.pickle_data(c.vehicles_preprocessed_train_data_p, vehicles)
        utils.pickle_data(c.non_vehicles_preprocessed_train_data_p, non_vehicles)
    else:
        vehicles = utils.unpickle_data(c.vehicles_preprocessed_train_data_p)
        non_vehicles = utils.unpickle_data(c.non_vehicles_preprocessed_train_data_p)

    X = np.concatenate((vehicles, non_vehicles))
    y = np.concatenate((np.ones(len(vehicles)), np.zeros(len(non_vehicles))-1))
    return X,y

def build_model(input_shape=(64,64,3), dropout=0.2, with_flatten=True):
    """create keras model with optional flatten layer"""
    model = Sequential()
    model.add(Convolution2D(8, (3,3), input_shape=input_shape, activation='elu', padding='same'))
    model.add(Convolution2D(16, (3,3), activation='elu', padding='same'))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(dropout))
    model.add(Convolution2D(128, (8,8), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(1, (1,1), activation='tanh'))
    if with_flatten:
        model.add(Flatten())
    return model

def find_vehicles_image(model, img_bgr, scales = [1.0], y_from=400, y_to=656):
    """detect vehicle areas in image"""
    img = img_bgr[y_from:y_to,:,:]
    img = utils.img_bgr2luv(img)
    img = img_normalize(img)

    img_w = img.shape[1]
    img_h = img.shape[0]

    detections = []

    for scale in scales:
        if not scale == 1:
            img_scaled = cv2.resize(img, (int(img_w/scale), int(img_h/scale)))
        else:
            img_scaled = img

        # print(img_scaled.shape)

        p = model.predict(img_scaled[None,:,:,:])

        # show the output
        # from pylab import rcParams
        # rcParams['figure.figsize'] = 12.8, 7.2
        # plt.imshow(p[0,:,:,0], cmap='hot')
        # plt.show()

        # get indices where output is hot
        y,x = np.nonzero(p[0,:,:,0]>0.995)

        step=8*scale
        window_size=64*scale

        for i,j in zip(x*step,y*step):
            # cv2.rectangle(img_cropped, (int(i),int(j)), (int(i+window_size),int(j+window_size)), (0,0,255), 1)
            detections.append([(int(i),int(j)+y_from), (int(i+window_size),int(j+window_size)+y_from)])

    return detections

def draw_detections(img_bgr, detections):
    draw_img = img_bgr.copy()
    for d in detections:
        cv2.rectangle(draw_img, d[0], d[1], (0,0,255), 1)
    return draw_img

def draw_heatmap(img_bgr, detections, threshold=8, frame_number=None, all_detections=None, n_frames=None):
    """add detections to heatmap, average accross n_frames and draw labeled areas"""
    hm = HeatMap(img_bgr.shape[:2])

    if frame_number is None:
        hm.add(detections)
    else:
        for o in range(max(0,frame_number-n_frames),frame_number):
            hm.add(all_detections[o])

    hm.threshold(threshold)
    hm.label()
    return hm.draw_labeled_bboxes(img_bgr.copy(), min_w=64, min_h=64)

def find_vehicles_folder(model, input_folder, output_folder, start_frame=0, end_frame=None, scales = [1.0], y_from=400, y_to=656):
    """find vehicles in a folder of images"""
    utils.make_dir(output_folder)

    all_detections = utils.unpickle_data(c.dl_detections_p)
    if all_detections is None:
        all_detections = []

    video_glob = glob.glob('{}/*.png'.format(input_folder))
    video_glob.sort()
    if end_frame is None:
        end_frame = len(video_glob)

    start_time = time.time()

    utils.print_progress_bar (0, len(video_glob), prefix = 'dl detection:')

    for current_frame in range(start_frame, end_frame):
        video_image_name = video_glob[current_frame]
        video_image_basename = os.path.basename(video_image_name)

        elapsed = (time.time() - start_time)/60
        eta = 0
        if current_frame>0:
            eta = (elapsed/current_frame)*(len(video_glob)-current_frame)

        utils.print_progress_bar(current_frame, len(video_glob),
                                 prefix = 'dl detection:',
                                 suffix='frame: {:d} - elapsed: {:.2f}min - eta: {:.2f}min'.format(current_frame,elapsed,eta))

        img_bgr = cv2.imread(video_image_name)

        detections = find_vehicles_image(model, img_bgr, scales=scales, y_from=y_from, y_to=y_to)

        draw_img = draw_detections(img_bgr, detections)
        # draw averaged, thresholded heatmap
        hmap_img = draw_heatmap(img_bgr, detections, threshold=16, frame_number=current_frame, all_detections=all_detections, n_frames=5)

        if len(all_detections)<=current_frame:
            all_detections.append(detections)
        else:
            all_detections[current_frame]=detections

        cv2.imwrite('{}/img{}'.format(output_folder, video_image_basename), draw_img)
        cv2.imwrite('{}/hmp{}'.format(output_folder, video_image_basename), hmap_img)

        current_frame += 1

    utils.print_progress_bar (len(video_glob), len(video_glob), prefix = 'dl detection:', suffix='{:.2f}min'.format((time.time()-start_time)/60))

    # save detections for debugging
    utils.pickle_data(c.dl_detections_p, all_detections)

def test_run(model, img_bgr, show_result=True, save_false=False, false_threshold=5):
    """test detection on single image, optionally display result and save false positives below threshold"""
    img_ori = img_bgr.copy()

    # detections = find_vehicles_image(model, img_bgr, scales = [1.0], y_from=400, y_to=656)

    start_time = time.time()
    detections = find_vehicles_image(model, img_bgr, scales = [1.0], y_from=0, y_to=720)
    print('detection time: {:.2f}'.format(time.time()-start_time))


    if detections is None:
        print('no vehicles detected')
        return

    for i in detections:
        cv2.rectangle(img_bgr,tuple(i[0]),tuple(i[1]),(0,0,255),1)

    hmap = HeatMap(img_bgr.shape[:2])
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
    # cv2.waitKey()

    if show_result:
        cv2.imshow('test', img_bgr)
        cv2.waitKey()

def model_summary_md(model):
    """print model summary in markdown format"""
    def get_layer_params(layer):
        c = layer.get_config()
        if l.__class__.__name__=='Conv2D':
            return 'depth: {}, stride: {}, activation: {}'.format(c['filters'], c['kernel_size'], c['activation'])
        elif l.__class__.__name__=='Dropout':
            return 'rate: {:.2f}'.format(c['rate'])
        elif l.__class__.__name__=='MaxPooling2D':
            return 'pool size: {}, stride: {}'.format(c['pool_size'], c['strides'])
        return ''
    print('| layer | parameters  | output shape |')
    print('| ----- | ----------- | ------------ |')
    for l in model.layers:
        print('|{}|{}|{}|'.format(l.__class__.__name__,get_layer_params(l),l.output_shape[1:]))
        # print(l.__class__.__name__)
        # print(l.get_config())

if __name__=='__main__':
    do_preprocess = True
    do_train = True
    do_test = False

    if do_train:
        print('training')
        X,y = load_train_data(preprocess=do_preprocess)
        print('train data:',X.shape, y.shape)

        model = build_model(with_flatten=True, dropout=0.3)

        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

        print(model.summary())
        model_summary_md(model)

        for i in range(16):
            print('epoch:',i)
            X,y = shuffle(X,y)
            model.fit(X, y, epochs=1, verbose=1, shuffle=True, validation_split=0.2)
        model.save_weights('model.h5')

    model = build_model(input_shape=(None, None, 3), with_flatten=False)
    model.load_weights('model.h5')

    print(model.summary())
    model_summary_md(model)

    if do_test:
        for i in range(961, 965):
            img_bgr = cv2.imread('{}/{:04d}.png'.format(c.project_video_images_folder, i))
            test_run(model, img_bgr, show_result=False, save_false=False, false_threshold=1)
        sys.exit(0)

    find_vehicles_folder(model, c.project_video_images_folder, c.output_folder_dl, start_frame=0, end_frame=None, scales = [0.8, 1.0, 1.5], y_from=400)




