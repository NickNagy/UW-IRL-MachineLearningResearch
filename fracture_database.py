import csv
import os
from scipy import io
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

FRACTURE_DIR = #
NON_FRACTURE_DIR = #
CSV_SAVE_PATH = #

debug = False
offset = 1  # matlab indexing starts @ 1
extension = ".jpg"

def writeLine(csvfile, img_path, x1, y1, x2, y2, class_name='fracture'):
    if class_name is None:
        csvfile.write(img_path + ",,,,,\n")
    else:
        csvfile.write(
            img_path + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + class_name + "\n")


def swap(a, b, cond):
    temp = b
    if cond:
        b = a
        a = temp
    return a, b

# TODO:
def write_file(arr, img_path, csvfile):
    for row in range(0, arr.shape[0]):
        if debug:
            print(img_path + ": " + str(arr[row]))
        else:
            x1, y1, w, h = tuple(arr[row])
            if not (w==0 or h==0):
            #x1, x2 = swap(x1, x2, x1 > x2)
            #y1, y2 = swap(y1, y2, y1 > y2)
                writeLine(csvfile, img_path, x1-offset, y1-offset, x1+w-offset, y1+h-offset)


def write_set(directory, csvfile, extension=".jpg"):
    for subdir, dirs, files in os.walk(directory):
        for folder in dirs:
            next_dir = directory + "\\" + folder
            os.chdir(next_dir)
            img_path = next_dir + "\\" + folder + extension
            try:
                write_file(np.array((io.loadmat('crop_image_fracturemask.mat'))['boundNow']), img_path, csvfile)
            except(FileNotFoundError):
                print("No bounds data. Assuming no fracture in image.")
                writeLine(csvfile, img_path, None, None, None, None, None)


def divide_and_write_fracture_sets(directory, csvfiles, train_ratio=0.7, validate_ratio=0.15, extension=".jpg"):
    # csvfiles is a dictionary w/ keys 'train', 'validate', 'test'
    for subdir, dirs, files in os.walk(directory):
        for folder in dirs:
            next_dir = directory + "\\" + folder
            os.chdir(next_dir)
            img_path = next_dir + "\\" + folder + extension
            which_set = random.uniform(0.0, 1.0)
            if which_set < train_ratio:
                wr = csvfiles['train']
            elif which_set >= train_ratio and which_set < train_ratio + validate_ratio:
                wr = csvfiles['validate']
            else:
                wr = csvfiles['test']
            try:
                write_file(np.array((io.loadmat('crop_image_fracturemask.mat'))['boundNow']), img_path, wr)
            except FileNotFoundError:
                print("File(s) missing. Ignoring folder " + str(folder))

def crop(img, coords, region, percent=0.8):
    h,w = np.shape(img)
    print(w)
    print(h)
    num_coords = coords.shape[0]
    new_coords = np.copy(coords)
    if region == 1: # good
        new_coords[:,2] = np.minimum(int(percent*w)*np.ones(num_coords), new_coords[:,2])
        new_coords[:,3] = np.minimum(int(percent*h)*np.ones(num_coords), new_coords[:,3])
        return img[0:int(percent * h), 0:int(percent * w)], new_coords
    if region == 2: # good
        new_coords[:,1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,1], int(0.5*(1-percent)*h)))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int(0.5 * (1 - percent) * h))#)
        return img[int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h), 0:int(percent * w)], new_coords
    if region == 3: # good
        new_coords[:,1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,1], int((1-percent)*h))) # good
        new_coords[:,3] = np.subtract(new_coords[:,3], int((1-percent)*h)) # good
        return img[int((1 - percent) * h):h, 0:int(percent * w)], new_coords
    if region == 4: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int(0.5*(1-percent)*w)))
        new_coords[:, 2] = np.minimum(int(0.5 * (percent) * w) * np.ones(num_coords),
                                      np.subtract(new_coords[:, 2], int(0.5 * (1 - percent) * w)))
        new_coords[:, 3] = np.minimum(np.ones(num_coords)*int(percent*h), new_coords[:,3])
        return img[0:int(percent * h), int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w)], new_coords
    if region == 5: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 0], int(0.5 * (1 - percent) * w)))
        new_coords[:, 1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 1], int(0.5 * (1 - percent) * h)))
        new_coords[:, 2] = np.minimum(int(0.5 * (percent) * w) * np.ones(num_coords), np.subtract(new_coords[:, 2],
                                                                                                  int(0.5 * (
                                                                                                  1 - percent) * w)))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int(0.5 * (1 - percent) * h))
        return img[int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h),
               int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w)], new_coords
    if region == 6: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int(0.5*(1-percent)*w)))
        new_coords[:, 1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 1], int((1 - percent) * h)))
        new_coords[:, 2] = np.minimum(int(0.5*(percent)*w)*np.ones(num_coords), np.subtract(new_coords[:,2],
                                                                                            int(0.5*(1-percent)*w)))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int((1 - percent) * h))
        return img[int((1 - percent) * h):h, int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w)], new_coords
    if region == 7: # good
        new_coords[:,0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int((1-percent)*w))) # good
        new_coords[:,2] = np.subtract(new_coords[:,2], int((1-percent)*w)) # good
        return img[0:int(percent * h), int((1 - percent) * w):w], new_coords
    if region == 8: # good
        new_coords[:, 0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 0], int((1 - percent) * w)))
        new_coords[:, 1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:, 1], int(0.5 * (1 - percent) * h)))
        new_coords[:, 2] = np.subtract(new_coords[:, 2], int((1 - percent) * w))
        new_coords[:, 3] = np.subtract(new_coords[:, 3], int(0.5 * (1 - percent) * h))  # )
        return img[int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h), int((1 - percent) * w):w], new_coords
    if region == 9: # good
        new_coords[:,0] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,0], int((1-percent)*w))) # good
        new_coords[:,1] = np.maximum(np.zeros(num_coords), np.subtract(new_coords[:,1], int((1-percent)*h))) # good
        new_coords[:,2] = np.subtract(new_coords[:,2], int((1-percent)*w)) # good
        new_coords[:,3] = np.subtract(new_coords[:,3], int((1-percent)*h)) # good
        return img[int((1 - percent) * h):h, int((1 - percent) * w):w], new_coords

def show_img_with_fractions(img, orig_coords, cropped_img=None, cropped_coords=None):
    if cropped_img is not None:
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(cropped_img)
        for (x1, y1, x2, y2) in orig_coords:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', fill=False)
            ax[0].add_patch(rect)
        for (x1, y1, x2, y2) in cropped_coords:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', fill=False)
            ax[1].add_patch(rect)
    else:
        print("This shouldn't be running")
        fig,ax = plt.subplots(1)
        ax.imshow(img)
        for (x1,y1,x2,y2) in orig_coords:
            rect = Rectangle((x1,y1),x2-x1, y2-y1, edgecolor='r', fill=False)
            ax.add_patch(rect)
    plt.show()

def test_crop_results(img, orig_coords, region, percent):
    print(orig_coords)
    cropped_img, cropped_coords = crop(img, orig_coords, region, percent)
    print(cropped_coords)
    show_img_with_fractions(img, orig_coords, cropped_img=cropped_img, cropped_coords=cropped_coords)

from PIL import Image

csv = open(CSV_SAVE_PATH + '\\new_validation_' + extension[1:] + '.csv', 'r')
img_path, x1, y1, x2, y2, _ = csv.readline().split(',')
orig_coords = np.array([[int(x1), int(y1), int(x2), int(y2)]])
while True:
    new_img_path, x1, y1, x2, y2, _ = csv.readline().split(',')
    if new_img_path != img_path:
        break
    orig_coords = np.append(orig_coords, [[int(x1), int(y1), int(x2), int(y2)]], axis=0)

if __name__ == '__main__':
    TL = 1
    ML = 2
    BR = 3
    TM = 4
    MM = 5
    BM = 6
    TR = 7
    MR = 8
    BR = 9

    img = np.asarray(Image.open(img_path).convert('L'))
    plt.gray()
    test_crop_results(img, orig_coords, MM, 0.9)

'''
training_csvfile = open(CSV_SAVE_PATH + '\\training_' + extension[1:] + '.csv', 'w')
validation_csvfile = open(CSV_SAVE_PATH + '\\validation_' + extension[1:] + '.csv', 'w')
testing_csvfile = open(CSV_SAVE_PATH + '\\testing_' + extension[1:] + '.csv', 'w')

one_image_csvfile = open(CSV_SAVE_PATH + '\\one_image.csv', 'w')
one_image_dir = 'C:\\Users\\Nick Nagy\\UW\\Mike D Bindschadler - RibFracture\\FracturePresent\\anon_images\\rib0010_1_2'
os.chdir(one_image_dir)
one_image_path = one_image_dir + '\\rib0010_1_2.jpg'
write_file(np.array((io.loadmat('crop_image_fracturemask.mat'))['boundNow']), one_image_path, one_image_csvfile)
one_image_csvfile.close()
'''
#csvfiles = {'train': training_csvfile, 'validate': validation_csvfile, 'test': testing_csvfile}

#divide_and_write_fracture_sets(FRACTURE_DIR, csvfiles, extension=extension)
#write_set(NON_FRACTURE_DIR + "\\Training", training_csvfile, extension=extension)
#write_set(NON_FRACTURE_DIR + "\\Validation", validation_csvfile, extension=extension)
#write_set(NON_FRACTURE_DIR + "\\Testing", testing_csvfile, extension=extension)

#training_csvfile.close()
#validation_csvfile.close()
#testing_csvfile.close()

