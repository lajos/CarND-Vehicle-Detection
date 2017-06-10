import numpy as np
import cv2
from scipy.ndimage.measurements import label
import utils
import constants as c
import matplotlib.pyplot as plt


class HeatMap:
    def __init__(self, shape):
        self.heatmap = np.zeros(shape)
        self.labels = None

    def add(self, bbox_list):
        """add detections to heatmap"""
        if bbox_list is None:
            return
        for box in bbox_list:
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def threshold(self, threshold):
        """discard heatmap areas below threshold"""
        self.heatmap[self.heatmap <= threshold] = 0

    def label(self):
        """label seperate areas in heatmap"""
        self.labels = label(self.heatmap)

    def get_clipped(self):
        """clip heatmap values to [0..255] range"""
        return np.clip(self.heatmap, 0, 255).astype(np.uint8)

    def draw_labeled_bboxes(self, img, min_w=85, min_h=65):
        """draw bounding boxes of labelled areas, discarding areas below min_w width and min_h height"""
        for car_number in range(1, self.labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # filter small detections
            if abs(bbox[0][0]-bbox[1][0])<min_w:
                continue
            if abs(bbox[0][1]-bbox[1][1])<min_h:
                continue

            if bbox[0][0]<img.shape[1]/2:
                continue

            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 2)
        return img

if __name__=='__main__':
    img_bgr = cv2.imread('{}/0502.png'.format(c.project_video_images_folder))
    img = img_bgr.copy()

    # cv2.imshow('test', img_bgr)
    # cv2.waitKey()

    heatmap = HeatMap(img_bgr.shape[:2])

    detections = utils.unpickle_data(c.test_detections_p)

    print(detections.shape)

    heatmap.add(detections)
    # heatmap.threshold(1)
    heatmap.label()

    clipped = heatmap.get_clipped()

    # cv2.imshow('test', clipped)
    # cv2.waitKey()

    # Find final boxes from heatmap using label function
    draw_img = heatmap.draw_labeled_bboxes(np.copy(img))

    draw_img = utils.img_reverse_channels(draw_img)

    fig = plt.figure(figsize=(20, 10), dpi=72)
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Bounding Box')
    plt.subplot(122)
    plt.imshow(clipped, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()
    cv2.waitKey()
