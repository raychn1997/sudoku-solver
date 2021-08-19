# %%
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import torch
from utility.torch import Model


def find_puzzle(image):
    # extract the puzzle square from an image

    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    return (puzzle, warped)


def extract_digit(cell):
    # extract the digits from a cell

    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # return the digit to the calling function
    return digit


def read_image(uploaded_file):
    # Read an image and return sudoku data
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    (puzzle, warped) = find_puzzle(image)
    puzzle_len_x, puzzle_len_y = warped.shape
    cell_len_x, cell_len_y = np.floor(puzzle_len_x/9), np.floor(puzzle_len_y/9)
    indices = []
    images = []

    # Go over each cell
    for r in range(1, 10):
        for c in range(1, 10):
            x_range_1 = int((r - 1) * cell_len_x)
            x_range_2 = int(r * cell_len_x)
            y_range_1 = int((c - 1) * cell_len_y)
            y_range_2 = int(c * cell_len_y)
            cell = warped[x_range_1:x_range_2, y_range_1:y_range_2]
            cell = extract_digit(cell)

            # None means no number
            if not cell is None:
                # Zoom in the digit
                # x,y,w,h = cv2.boundingRect(cell)
                # cell = cell[int(y*0.9):int((y+h)*1.1), int(x*0.9):int((x+w)*1.1)]

                # Resize the image to MNIST's dimension
                cell = cv2.resize(cell, (28, 28))/255
                indices.append((r, c))
                images.append(cell)

    # Run the model over all the cells that have numbers
    model = Model.load_from_checkpoint('model/model.ckpt')
    model.eval()
    x = np.array(images)
    x = torch.from_numpy(x.astype(np.float32))
    logits = model(x.reshape(-1, 1, 28, 28))
    y_hat = torch.argmax(logits, dim=1).detach()

    # Convert the result to data points that are used by the solver
    data = []
    for i, indice in enumerate(indices):
        data.append((indice[0], indice[1], int(y_hat[i])))
    return data


