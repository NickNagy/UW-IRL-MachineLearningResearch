# Contents of this file have changed, and function descriptions are not up to date - NN

import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.ndimage import rotate
from matplotlib import pyplot as plt

BATCH_DIR = #
#BATCH_DIR = #
SAVE_DIR = BATCH_DIR

OUTPUT_WIDTH = 128
OUTPUT_HEIGHT = OUTPUT_WIDTH

debug = 0  # 1 if display data, 0 if save data
SHOW_DEBUG_DISPLAYS = False

#suffix = "0p3" # no longer used

X_NAME = r'crop_image.mat'
GT_NAME = r'labelIm.*'
WEIGHT_NAME = r'boundaryWeightIm.*'

FORCE_Y_OVERWRITE = True # force overwrite of multilayer Y images (because of earlier error)

BACKGROUND = 0
SPINE = 1
MEDIASTINUM = 2
LEFT_LUNG = 3
RIGHT_LUNG = 4
LEFT_SUBDIAPHRAGM = 5
RIGHT_SUBDIAPHRAGM = 6
CARINA = 7

REGION = BACKGROUND


# params:
#   img_in: input np array to extract region from
#   val: pixel value to identify region from img_in
#   output width: width of returned np array
#   output height: height of returned np array
# returns:
#   an np array highlighting the region in img_in w/ pixel value 'val'
#   ; dimensions = (output_width, output_height)
def resize_region(img_in, val, output_width, output_height):
    img_in_resize = imresize(img_in, (output_width, output_height))
    regionArr = np.ndarray(shape=(output_width, output_height, 2), dtype='uint8')
    for i in range(0, 2):
        layerArr = np.ndarray(shape=(output_width, output_height), dtype='uint8')
        for j in range(0, output_width):
            for k in range(0, output_height):
                if i == 0:
                    if img_in_resize[j, k] == 0:
                        layerArr[j, k] = 1
                    else:
                        layerArr[j, k] = 0
                else:
                    if img_in_resize[j, k] == val:
                        layerArr[j, k] = int(val > 0)
                    else:
                        layerArr[j, k] = int(val == 0)
        regionArr[:, :, i] = layerArr
    return regionArr


# params:
#   img_in: input np array to extract region from
#   n_class: number of classes of regions in img_in
#   output width: width of returned np array
#   output height: height of returned np array
# returns:
#   an np array with each layer highlighting a specific class of pixel values from img_in
#   ; dimensions = (output_width, output_height, n_class)
def resize_multilayer(img_in, output_width, output_height, vals=[BACKGROUND, SPINE, MEDIASTINUM, LEFT_LUNG, RIGHT_LUNG,
                                                                 LEFT_SUBDIAPHRAGM, RIGHT_SUBDIAPHRAGM, CARINA]):

    # Resize Image using nearest neighbor interp
    img_in_resize = imresize(img_in, (output_width, output_height), interp="nearest")
    # The single-pixel carina label was probably lost in the resize, need to 
    # ensure that one pixel is assigned to the carina
    if CARINA in vals:
        temp_nz = np.nonzero(img_in == CARINA)
        input_width, input_height = np.shape(img_in)
        new_carina_loc = (
            int(temp_nz[0][0] * output_width / input_width), 
            int(temp_nz[1][0] * output_height / input_height))
        if debug: print("Carina location: " + str(new_carina_loc))
    img_in_resize[new_carina_loc[0],new_carina_loc[1]] = CARINA
    
    n_class = len(vals)
    multilayerArr = np.ndarray(shape=(output_width, output_height, n_class), dtype='bool')
    
    for val in vals:
        mask = img_in_resize==val
        multilayerArr[:,:,val] = mask
        # NB that this code will assign each layer according to the numerical order of 
        # regions, even if the order they are listed in "vals" changes.  Note also
        # that it will probably cause problems if the range of values is not exactly
        # the integers between 0 and n_class-1 because it will try to assign values
        # to an index of multilayerArr which is outside its size
        
        # The old way of constructing the multi-layer array made the carina point usually
        # have two labels (which is not what we wanted). Furthermore, because argmax
        # takes the first maximum and the carina is the last layer, the carina label
        # was still being completely ignored in scoring (though not in training, I think)
    
    # multilayerArr used to be a 'uint8' array, but really ought to be a logical 
    # array.  I've rewritten that above, but I'm not sure if that will break something
    # in our already trained networks.  If so, then it could be cast back to a uint8
    # before being returned here
    return multilayerArr


# params:
#   x: an original image np array
#   y: a ground truth np array from x
# returns: None
#
# makes subplots of x and y images
def display_arrays(x, y, w=None):
    plt.gray()

    if len(np.shape(y)) > 2:
        num_classes = np.shape(y)[2]
        print(np.shape(y))
        print(num_classes)
        fig, ax = plt.subplots(2 + int(w is not None), num_classes, sharex=True, sharey=True)
        ax[0, 0].imshow(x, aspect="auto")
        for i in range(0, num_classes):
            ax[1, i].imshow(y[:, :, i], aspect="auto")
        if w is not None:
            ax[2, 0].imshow(w, aspect="auto")
    else:
        fig, ax = plt.subplots(1, 2 + int(w is not None), sharex=True, sharey=True)
        ax[0].imshow(x, aspect="auto")
        ax[1].imshow(y, aspect="auto")
        if w is not None:
            ax[2].imshow(w, aspect="auto")
    plt.show()


# params:
#   img_in: image to crop
#   region: valid arguments = {"TL", "TM", "TR", "ML", "MM", "MR", "BL", "BM", "BR"}
#   percent: percentage of original image remaining (a percentage = 0.67 will crop 2/3 of the original)
def crop_region(img_in, region, percent=0.8):
    if img_in is None: return
    w, h = np.shape(img_in)
    if region == "TL":
        return img_in[0:int(percent * w), 0:int(percent * h)]
    if region == "ML":
        return img_in[int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w), 0:int(percent * h)]
    if region == "BL":
        return img_in[int((1 - percent) * w):w, 0:int(percent * h)]
    if region == "TM":
        return img_in[0:int(percent * w), int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h)]
    if region == "MM":
        return img_in[int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w),
               int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h)]
    if region == "BM":
        return img_in[int((1 - percent) * w):w, int(0.5 * (1 - percent) * h):int(0.5 * (1 + percent) * h)]
    if region == "TR":
        return img_in[0:int(percent * w), int((1 - percent) * h):h]
    if region == "MR":
        return img_in[int(0.5 * (1 - percent) * w):int(0.5 * (1 + percent) * w), int((1 - percent) * h):h]
    if region == "BR":
        return img_in[int((1 - percent) * w):w, int((1 - percent) * h):h]


# params:
#   num_rotations: how many rotations cw AND ccw. Each rotation is 10 degrees
#   include_rotations: TRUE if want to resize/save rotated versions of input data
#   output_width: width of output
#   output_height: height of output
#   file_name: name of file
# returns: None
def rotate_and_resize(x, gt, w, file_name, crop_type="none", num_rotations=9, include_rotations=0,
                      output_width=OUTPUT_WIDTH, output_height=OUTPUT_HEIGHT):
    # print("resizing unrotated...")
    xResize = imresize(x, (output_width, output_height))
    gtResize = resize_multilayer(gt, output_width=output_width,
                                 output_height=output_height)  # resize_region(gt, val=BACKGROUND, output_width=output_width, output_height=output_height)#resize_multilayer(img_in=gt, output_width=output_width, output_height=output_height)

    wResize = np.ones((output_width, output_height))

    if w is not None:
        wResize = imresize(w, (output_width, output_height))

    # TODO: update to display weight layer
    if debug:
        if SHOW_DEBUG_DISPLAYS:
            print("displaying...")
            display_arrays(x=xResize, y=gtResize, w=wResize)
    else:
        print("saving unrotated...")
        x_save_name = file_name + "_" + crop_type + "_0_x.npy"
        existingFiles = os.listdir(SAVE_DIR)
        # If saved already, don't overwrite for orig images
        if x_save_name in existingFiles:
            print('resized orig file already saved, not overwriting...')
        else:
            print('saving resized orig image...')
            np.save(os.path.join(SAVE_DIR, x_save_name), xResize)
            
        y_save_name = file_name + "_" + crop_type + "_0_y.npy"
        if FORCE_Y_OVERWRITE or ~(y_save_name in existingFiles):
            print('saving label array...')
            np.save(os.path.join(SAVE_DIR, y_save_name), gtResize)
        else:
            print('label array already saved, not overwriting...')

        w_save_name = file_name + "_" + crop_type + "_0_w.npy"
        if w_save_name in existingFiles:
            print('resized weight image already saved, not overwriting...')
        else:
            np.save(os.path.join(SAVE_DIR, w_save_name), wResize)

    for i in range(1, (include_rotations * (num_rotations + 1))):
        print("rotating by " + str(i * 10) + " degrees...")
        print("rotating by " + str(360 - (i * 10)) + " degrees...")
        x_ccw = rotate(xResize, i * 10, reshape=False)
        x_cw = rotate(xResize, 360 - (i * 10), reshape=False)

        num_classes = np.shape(gtResize)[2]
        gt_ccw = np.ndarray(shape=(output_width, output_height, num_classes))
        gt_cw = np.ndarray(shape=(output_width, output_height, num_classes))

        if debug:
            print("GT CCW SHAPE: ", np.shape(gt_ccw))
            print("Rotated shape: ", np.shape(rotate(gtResize, i * 10, cval=1)))

        gt_ccw[:, :, 0] = rotate(gtResize[:, :, 0], i * 10, cval=1, reshape=False)
        gt_cw[:, :, 0] = rotate(gtResize[:, :, 0], 360 - (i * 10), cval=1, reshape=False)
        for layer in range(1, num_classes):
            gt_ccw[:, :, layer] = rotate(gtResize[:, :, layer], i * 10, reshape=False)
            gt_cw[:, :, layer] = rotate(gtResize[:, :, layer], 360 - (i * 10), reshape=False)

        w_cval = 1
        if w is not None:
            w_cval = 15

        w_ccw = rotate(wResize, angle=i * 10, axes=(0, 1), reshape=False, cval=w_cval)
        w_cw = rotate(wResize, angle=360 - (i * 10), axes=(0, 1), reshape=False, cval=w_cval)

        if debug:
            print("displaying...")
            display_arrays(x=x_ccw, y=gt_ccw, w=w_ccw)
            display_arrays(x=x_cw, y=gt_cw, w=w_cw)
        else:
            print("saving rotated...")
            np.save(os.path.join(SAVE_DIR, file_name + "_" + str(i * 10) + "_" + crop_type + "_x.npy"), x_ccw)
            np.save(os.path.join(SAVE_DIR, file_name + "_" + str(360 - (i * 10)) + "_" + crop_type + "_x.npy"), x_cw)
            np.save(os.path.join(SAVE_DIR, file_name + "_" + str(i * 10) + "_" + crop_type + "_y.npy"), gt_ccw)
            np.save(os.path.join(SAVE_DIR, file_name + "_" + str(360 - (i * 10)) + "_" + crop_type + "_y.npy"), gt_cw)
            np.save(os.path.join(SAVE_DIR, file_name + "_" + str(i * 10) + "_" + crop_type + "_w.npy"), w_ccw)
            np.save(os.path.join(SAVE_DIR, file_name + "_" + str(360 - (i * 10)) + "_" + crop_type + "_w.npy"), w_cw)


# params:
#
# returns: None
def augmentations(x, gt, w, pad, file_name, include_rotations=0, include_cropping=0):
    file_name = file_name + "_" + str(pad)
    rotate_and_resize(x=x, gt=gt, w=w, file_name=file_name, include_rotations=include_rotations)
    if include_cropping:
        print("cropping data...")
        rotate_and_resize(x=crop_region(img_in=x, region="TL"), gt=crop_region(img_in=gt, region="TL"),
                          w=crop_region(img_in=w, region="TL"), file_name=file_name + "_TL_")
        rotate_and_resize(x=crop_region(img_in=x, region="TM"), gt=crop_region(img_in=gt, region="TM"),
                          w=crop_region(img_in=w, region="TM"), file_name=file_name + "_TM_")
        rotate_and_resize(x=crop_region(img_in=x, region="TR"), gt=crop_region(img_in=gt, region="TR"),
                          w=crop_region(img_in=w, region="TR"), file_name=file_name + "_TR_")
        rotate_and_resize(x=crop_region(img_in=x, region="ML"), gt=crop_region(img_in=gt, region="ML"),
                          w=crop_region(img_in=w, region="ML"), file_name=file_name + "_ML_")
        rotate_and_resize(x=crop_region(img_in=x, region="MM"), gt=crop_region(img_in=gt, region="MM"),
                          w=crop_region(img_in=w, region="MM"), file_name=file_name + "_MM_")
        rotate_and_resize(x=crop_region(img_in=x, region="MR"), gt=crop_region(img_in=gt, region="MR"),
                          w=crop_region(img_in=w, region="MR"), file_name=file_name + "_MR_")
        rotate_and_resize(x=crop_region(img_in=x, region="BL"), gt=crop_region(img_in=gt, region="BL"),
                          w=crop_region(img_in=w, region="BL"), file_name=file_name + "_BL_")
        rotate_and_resize(x=crop_region(img_in=x, region="BM"), gt=crop_region(img_in=gt, region="BM"),
                          w=crop_region(img_in=w, region="BM"), file_name=file_name + "_BM_")
        rotate_and_resize(x=crop_region(img_in=x, region="BR"), gt=crop_region(img_in=gt, region="BR"),
                          w=crop_region(img_in=w, region="BR"), file_name=file_name + "_BR_")


n_class = 8

include_weights = 1
include_rotations = 0
include_cropping = 0
include_padding = 0 # related to augmentations, not to convolutions

print("Starting...")

import os
#import matlab.engine
#import scipy
from scipy import io # for io.loadmat
import re # regular expressions

#eng = matlab.engine.start_matlab()

for subdir, dirs, files in os.walk(BATCH_DIR):
    for folder in dirs:
        next_dir = BATCH_DIR + "\\" + folder

        os.chdir(next_dir)
        #eng.cd(next_dir)

        name = folder
        print("FOLDER: " + folder)

        skip = 0
        initials = 'NN'

        print("retrieving GT image...")
        
        # Find list of files which match the pattern for the GT image name
        fileList = [f for f in os.listdir() if re.match(GT_NAME,f)]
        if not fileList: #list is empty, file not found
            print("GT image wasn't found. Skipping folder...")
            skip = 1
        else:
            # Throw error if there is more than one image found
            assert len(fileList)==1, 'Multiple GT images found!'
            # Load the image and put it in an array
            gt = np.asarray(Image.open(fileList[0]))
                   

        if not skip:
            w = None

            if include_weights:
                print("retrieving weight image...")
                fileList = [f for f in os.listdir() if re.match(WEIGHT_NAME,f)]
                if not fileList:
                    print("weight image wasn't found. Skipping folder...")
                    skip = 1
                else:
                    # Throw error if there is more than one image found
                    assert len(fileList)==1, 'Multiple weight images found!'
      
        if not skip:
            print("retrieving x image...")
            # check that an x image exists
            try:
                #  Load from matlab file
                x = np.array((io.loadmat(X_NAME,appendmat=False))['dxImage']['img'][0][0])
                # Explaining previous line:  The original image is in 'crop_image.mat'
                # In that mat file there is a variable called dxImage which is a struct
                # That struct has a the image in a field named img. 
                # The python inport of this ends up putting it in an array of type object
                # inside of a nested list at [0][0] (not sure why this happens)
                # It's not clear to me whether the outermost np.array call is necessary,
                # but it shouldn't hurt 
                
                # Below line: The old way of doing this, using a full-on MATLAB engine
                # x = np.asarray(eng.getfield(eng.load(X_NAME)['dxImage'], 'img'))
            except FileNotFoundError:
                print("x image wasn't found. Skipping folder...")
                skip = 1

        if not skip:
            augmentations(x=x, gt=gt, w=w, pad="0", file_name=name, include_rotations=include_rotations,
                          include_cropping=include_cropping)

            if include_padding:
                print("augmentations for first pad...")
                augmentations(x=np.pad(x, (1000, 1000), 'constant', constant_values=0),
                              gt=np.pad(gt, (1000, 1000), 'constant', constant_values=0),
                              w=np.pad(w, (1000, 1000), 'constant', constant_values=10), pad="1",
                              file_name=name,
                              include_rotations=include_rotations,
                              include_cropping=include_cropping)
print('Done!!')
