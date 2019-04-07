from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import time
from PIL import Image
from scipy.misc import imread
import imageio
import csv
from sklearn.metrics import jaccard_similarity_score
from collections import defaultdict

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image, apply_transform#, crop_image
from keras_retinanet.utils.transform import transform_aabb
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import tensorflow as tf

import ast

def get_session():
    config = tf.ConfigProto()#device_count = {'GPU': 0}) # run on CPU?
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

def IoU(true, pred):
    xA = np.maximum(pred[0], np.transpose(true[0]))
    yA = np.maximum(pred[1], np.transpose(true[1]))
    xB = np.minimum(pred[2], np.transpose(true[2]))
    yB = np.minimum(pred[3], np.transpose(true[3]))
    interArea = np.maximum((xB-xA), 0)*np.maximum((yB-yA),0)
    pred_area = (pred[2]-pred[0])*(pred[3]-pred[1])
    truth_area = (true[2]-true[0])*(true[3]-true[1])
    iou_arr = interArea/(pred_area + np.transpose(truth_area)-interArea)
    return np.mean(iou_arr)

def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )
    return output


def crop_image(image, coords, region, percent=0.9):
    """
    :param image:
    :param coords:
    :param region: integers 1-9
    :param percent:
    :return:
    """
    h, w = np.shape(image)[:2]
    num_coords = coords.shape[0]
    new_coords = np.copy(coords)
    if region == 0:
        return image, new_coords
    if region == 1: # good
        new_coords[:,2] = np.minimum(int(percent*w)*np.ones(num_coords), new_coords[:,2])
        new_coords[:,3] = np.minimum(int(percent*h)*np.ones(num_coords), new_coords[:,3])
        return image[0:int(percent * h), 0:int(percent * w), :], new_coords
    if region == 2: # good
        new_coords[:,1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,1], int(0.5*(1-percent)*h)))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int(0.5 * (1 - percent) * h))#)
        return image[int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h), 0:int(percent * w), :], new_coords
    if region == 3: # good
        new_coords[:,1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,1], int((1-percent)*h))) # good
        new_coords[:,3] = np.subtract(new_coords[:,3], int((1-percent)*h)) # good
        return image[int((1 - percent) * h):h, 0:int(percent * w), :], new_coords
    if region == 4: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int(0.5*(1-percent)*w)))
        new_coords[:, 2] = np.minimum(int(0.5 * (percent) * w) * np.ones(num_coords),
                                      np.subtract(new_coords[:, 2], int(0.5 * (1 - percent) * w)))
        new_coords[:, 3] = np.minimum(np.ones(num_coords)*int(percent*h), new_coords[:,3])
        return image[0:int(percent * h), int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w), :], new_coords
    if region == 5: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 0], int(0.5 * (1 - percent) * w)))
        new_coords[:, 1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 1], int(0.5 * (1 - percent) * h)))
        new_coords[:, 2] = np.minimum(int(0.5 * (percent) * w) * np.ones(num_coords), np.subtract(new_coords[:, 2],
                                                                                                  int(0.5 * (
                                                                                                  1 - percent) * w)))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int(0.5 * (1 - percent) * h))
        return image[int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h),
               int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w)], new_coords
    if region == 6: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int(0.5*(1-percent)*w)))
        new_coords[:, 1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 1], int((1 - percent) * h)))
        new_coords[:, 2] = np.minimum(int(0.5*(percent)*w)*np.ones(num_coords), np.subtract(new_coords[:,2],
                                                                                            int(0.5*(1-percent)*w)))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int((1 - percent) * h))
        return image[int((1 - percent) * h):h, int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w),:], new_coords
    if region == 7: # good
        new_coords[:,0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int((1-percent)*w))) # good
        new_coords[:,2] = np.subtract(new_coords[:,2], int((1-percent)*w)) # good
        return image[0:int(percent * h), int((1 - percent) * w):w], new_coords
    if region == 8: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 0], int((1 - percent) * w)))
        new_coords[:, 1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 1], int(0.5 * (1 - percent) * h)))
        new_coords[:, 2] = np.subtract(new_coords[:, 2], int((1 - percent) * w))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int(0.5 * (1 - percent) * h))  # )
        return image[int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h), int((1 - percent) * w):w,:], new_coords
    if region == 9: # good
        new_coords[:,0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int((1-percent)*w))) # good
        new_coords[:,1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,1], int((1-percent)*h))) # good
        new_coords[:,2] = np.subtract(new_coords[:,2], int((1-percent)*w)) # good
        new_coords[:,3] = np.subtract(new_coords[:,3], int((1-percent)*h)) # good
        return image[int((1 - percent) * h):h, int((1 - percent) * w):w,:], new_coords
    return

def evaluate_image(model, img_path, save_path, labels_to_names={0: "no fracture", 1: "fracture"}, true_coords=None, IoU_threshold=0.8, score_threshold=0.6,
                   transformation_matrix=None, crop_type=0):
    image = read_image_bgr(img_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    if transformation_matrix is not None:
        image = apply_transform(transformation_matrix, image, None)
        if true_coords is not None:
            true_coords = transform_aabb(transformation_matrix, true_coords)

    #TODO
    if crop_type > 0:
        image, true_coords = crop_image(image, true_coords, crop_type)

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    #print("processing time: ", time.time()-start)

    boxes /= scale

    num_truth_w_threshold = 0
    num_pred_w_threshold = 0
    pos_sensitivity = 0
    num_true_coords = 0
    if true_coords is not None:
        coords_copy = true_coords.copy() # allows removing truth boxes that already have an associated prediction
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < score_threshold:
            break
        #color = label_color(label)
        color = (255,0,0) # red for bad prediction
        for i in range(len(coords_copy)):
            coords = coords_copy[i]
            if IoU(box, coords) > IoU_threshold:
                print("Prediction: " + str(box) + "; Truth: " + str(coords) + "; Score: " + str(score) + "; IoU: " + str(
                    IoU(box, coords)))
                num_truth_w_threshold += 1
                coords_copy = np.delete(coords_copy, i, 0)
                print(coords_copy)
                color = (0,255,0) # green for good prediction
                break
        b = box.astype(int)
        draw_box(draw, b, color=color)
        #caption = "{}{:.3f}".format(labels_to_names[label], score)
        #draw_caption(draw, b, caption)
        num_pred_w_threshold += 1
    for coords in true_coords:
        draw_box(draw, coords, color=(0,0,255)) # blue for true
    pos_sensitivity = num_truth_w_threshold / len(true_coords)
    num_true_coords = len(true_coords)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    plt.imshow(draw)
    plt.savefig(save_path + ".jpg")
    #plt.show()
    plt.close()
    try:
        pos_pred_val = num_truth_w_threshold / num_pred_w_threshold
    except ZeroDivisionError:
        pos_pred_val = 0
    return pos_sensitivity, pos_pred_val, num_truth_w_threshold, num_true_coords, num_pred_w_threshold

def preprocess_memfile(memfile):
    mem = defaultdict(dict)
    key = -1 # I don't know
    idx = 0
    while True:
        line = memfile.readline().split('\n')[0]
        if len(line) == 0:
            break
        if len(line.split(',')) == 1 and len(line.split(',')[0]) < 3: # epoch data
            key = line.split(',')[0]
            idx = 0
        else:
            mem[str(key)][str(idx)] = line.split(',')
            idx += 1
    return mem

def load_augmented_img(mem, epoch, idx):
    return mem[epoch][idx]

def convert_str_to_list(s):
    lst = ast.literal_eval(s)
    return [int(i.strip()) for i in lst]

def generate_image_metrics(model, csvfile):
    img, x1, y1, x2, y2, _ = csvfile.readline().split(',')
    true_coords = np.array([[int(x1), int(y1), int(x2), int(y2)]])
    sensitivities = []
    pos_pred_vals = []
    total_truth_w_threshold = 0
    total_true_boxes = 0
    total_pred_w_threshold = 0
    counter = 0
    while True:
        try:
            while True:
                newImg, x1, y1, x2, y2, _ = csvfile.readline().split(',')
                if newImg != img:
                    print(str(counter) + "\n Evaluating: " + str(img))
                    pos_sensitivity, pos_pred_val, num_truth_w_threshold, num_true_boxes, num_pred_w_threshold = \
                        evaluate_image(model, img, save_path=save_path+str(counter), true_coords=true_coords)
                    sensitivities.append(pos_sensitivity)
                    pos_pred_vals.append(pos_pred_val)
                    total_truth_w_threshold += num_truth_w_threshold
                    total_true_boxes += num_true_boxes
                    total_pred_w_threshold += num_pred_w_threshold
                    true_coords = np.array([[int(x1), int(y1), int(x2), int(y2)]])
                    img = newImg
                    counter += 1
                    break
                true_coords = np.append(true_coords, [[int(x1),int(y1),int(x2),int(y2)]], axis=0)
        except ValueError:
            print("broke")
            break
    print("Average sensitivity: " + str(sum(sensitivities)/len(sensitivities)) + "; Average predictive value: " +
          str(sum(pos_pred_vals)/len(pos_pred_vals)))
    print("Total number of ground truths with a prediction IoU > threshold: " + str(total_truth_w_threshold))
    print("Total number of ground truth boxes: " + str(total_true_boxes))
    print("Total number of prediction boxes: " + str(total_pred_w_threshold))
    print("Sensitivity calculated from box sum ratios: " + str(total_truth_w_threshold/total_true_boxes))
    try:
        print("Positive predictive value calculated from box sum ratios: " + str(total_truth_w_threshold/total_pred_w_threshold))
    except ZeroDivisionError:
        print("Positive predictive value calculated from box sum ratios: 0")

if __name__ == '__main__':
    save_path = "snapshots//LR0.0001LongRun//Image Predictions//50 Epoch//Validation//"
    model = models.load_model('snapshots//LR0.0001LongRun//resnet50_csv_50_inference.h5', backbone_name='resnet50')
    csvfile = open('new_validation_jpg.csv', 'r')
    img_path, x1, y1, x2, y2, _ = csvfile.readline().split(',')


    generate_image_metrics(model, csvfile)
