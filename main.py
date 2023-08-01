import os
from tqdm import tqdm
from glob import glob
import cv2
from run_agllnet import run

# set paths
INPUT_PATH = "input"
OUTPUT_PATH = "output"

WORK1_PATH = "work1"
WORK2_PATH = "work2"

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    original_size = (h, w)

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized, original_size

def resize(input_path, output_path, w, h):
    """Resize images to w x h. contain."""

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    path = glob(os.path.join(input_path, '*.*'))

    original_size = {}

    for i in tqdm(range(len(path))):
        img_A_path = path[i]
        img_A = cv2.imread(img_A_path)

        img_A, original_size[img_A_path] = image_resize(img_A, w, h)

        filename = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(img_A_path))[0])

        cv2.imwrite(filename + '.png', img_A)

    return original_size
        


def reresize(input_path, output_path, original_size):
    """Resize images to original size."""

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    path = glob(os.path.join(input_path, '*.*'))

    for i in tqdm(range(len(path))):
        img_A_path = path[i]
        img_A = cv2.imread(img_A_path)

        img_A = cv2.resize(img_A, original_size[img_A_path], interpolation=cv2.INTER_LANCZOS4)

        filename = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(img_A_path))[0])

        cv2.imwrite(filename + '.png', img_A)

if __name__ == "__main__":
    # resize images
    original_size = resize(INPUT_PATH, WORK1_PATH, 480, 480)

    # compute results
    run(WORK1_PATH, WORK2_PATH)

    # resize images
    reresize(WORK2_PATH, OUTPUT_PATH, original_size)