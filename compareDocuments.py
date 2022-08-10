import cv2
from skimage.metrics import structural_similarity as ssim
import imutils
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path


def pdf_jpg(path1, path2):
  p1 = convert_from_path(path1)
  p2 = convert_from_path(path2)
  p1[0].save('image_1'+ '.jpg', 'JPEG')
  p2[0].save('image_2'+ '.jpg', 'JPEG')

def sigExtract(inputPath, outputPath):
  # Load image and HSV color threshold
  image = cv2.imread(inputPath)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower = np.array([10, 10, 0])
  upper = np.array([145, 500, 255])
  mask = cv2.inRange(hsv, lower, upper)
  result = cv2.bitwise_and(image, image, mask=mask)
  result[mask==0] = (255,255,255)

  # Find contours on extracted mask, combine boxes, and extract ROI
  cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts = np.concatenate(cnts)
  x,y,w,h = cv2.boundingRect(cnts)
  cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
  # print(x, y, w, h)
  ROI = image[y:y+h, x:x+w]
  cv2.imwrite(outputPath, ROI)

#   cv2.imshow('Result', result)
#   cv2.imshow('Mask', mask)
#   cv2.imshow('Image', image)
  cv2.imshow('Extracted Signature', ROI)
  cv2.waitKey(0)


def match(path1, path2, countour = False):
    # read the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # turn images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # resize images for comparison
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300)) 

    (score, diff) = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")

    if countour == True:
      thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      
      for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
      # show the output images
      # cv2_imshow(img1)
      # cv2_imshow(img2)
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
      ax1.imshow(np.squeeze(img1), cmap='gray')
      ax2.imshow(np.squeeze(img2), cmap='gray')
      ax1.set_title('image1')
      ax2.set_title('image2')
      ax1.axis('off')
      ax2.axis('off')
      plt.show()

    similarity_value = "{:.2f}".format(ssim(img1, img2)*100)
    print("The two images are a " + str(float(similarity_value)) + "% match.")


pdf_jpg("F:/Downloads/genuine.pdf", "F:/Downloads/forged.pdf")

sigExtract('image_1.jpg', 'image_1.png')
sigExtract('image_2.jpg', 'image_2.png')

match('image_1.png', 'image_2.png', countour = True)