# import the necessary packages
import argparse
import glob

import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
                help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
                help="Flag indicating whether or not to visualize each iteration")
ap.add_argument("-b", "--threshold", type=float, default=0.8,
                help="threshold for multi-template matching")
args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
# cv2.imshow("Template", template)
# loop over the images to find the template in
print("[INFO] Starting Video Feed")
vid = cv2.VideoCapture(0)
while( True):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    web,image = vid.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1, 10)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        ret, resized = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        # edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (yCoords, xCoords) = np.where(result >= args["threshold"])
        clone = image.copy()
        # print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))
        edged = resized
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # check to see if the iteration should be visualized
        if args.get("visualize", False):
            # draw a bounding box around the detected region
            for (x, y) in zip(xCoords, yCoords):
                # draw the bounding box on the image
                cv2.rectangle(clone, (x, y), (x + tW, y + tH), (255, 0, 0), 3)
            # show our output image *before* applying non-maxima suppression
            # initialize our list of rectangles
            rects = []
            # loop over the starting (x, y)-coordinates again
            for (x, y) in zip(xCoords, yCoords):
                # update our list of rectangles
                rects.append((x, y, x + tW, y + tH))
            # apply non-maxima suppression to the rectangles
            pick = non_max_suppression(np.array(rects))
            # print("[INFO] {} matched locations *after* NMS".format(len(pick)))
            # loop over the final bounding boxes
            for (startX, startY, endX, endY) in pick:
                # draw the bounding box on the image
                cv2.rectangle(image, (startX, startY), (endX, endY),
                                (255, 0, 0), 3)
            # show the output image
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
