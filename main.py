import utils, histogram
import constants as c
import glob, pickle
import cv2
import numpy as np


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
    vehicle_images = make_image_pickle(c.vehicles_train_data_folder, c.vehicles_train_data_p, (64,64,3))
    non_vehicle_images = make_image_pickle(c.non_vehicles_train_data_folder, c.non_vehicles_train_data_p, (64,64,3))
    vehicle_hists = histogram.multispace_histograms_images(vehicle_images, c.vehicles_histograms_p)
    non_vehicle_hists = histogram.multispace_histograms_images(non_vehicle_images, c.non_vehicles_histograms_p)


if __name__=='__main__':
    preprocess()

    vehicle_images = utils.unpickle_data(c.vehicles_train_data_p)
    non_vehicle_images = utils.unpickle_data(c.non_vehicles_train_data_p)
    print('vehicle images: ',vehicle_images.shape)
    print('non_vehicle images: ',non_vehicle_images.shape)


    vehicle_hists = utils.unpickle_data(c.vehicles_histograms_p)
    non_vehicle_hists = utils.unpickle_data(c.non_vehicles_histograms_p)
    print('vehicle hists: ',vehicle_hists.shape)
    print('non_vehicle hists: ',non_vehicle_hists.shape)



