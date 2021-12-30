import sys
import cv2
import numpy as np

# from google.colab.patches import cv2_imshow

# Load our images
img1 = cv2.imread("1.png")
img2 = cv2.imread("2.png")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# cv2.imshow('img1', img1_gray)
# cv2.imshow('img2', img2_gray)

# Create our ORB detector and detect keypoints and descriptors
# orb = cv2.ORB_create(nfeatures=2000)
orb = cv2.SIFT_create(nfeatures=2000)

# Find the key points and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# cv2.imwrite('out1.png', cv2.drawKeypoints(img1, keypoints1, None, (255, 0, 255)))
# cv2.imwrite('out2.png', cv2.drawKeypoints(img2, keypoints2, None, (255, 0, 255)))

# Create a BFMatcher object.
# It will find all of the matching keypoints on two images
# bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
bf = cv2.BFMatcher_create(cv2.NORM_L1)

# Find matching points
matches = bf.knnMatch(descriptors1, descriptors2,k=2)

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2] 
    # Create a blank image with the size of the first image + second image
    output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    output_img[:r1, c:c+c1, :] = np.dstack([img2, img2, img2])  
    # Go over all of the matching points and extract them
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt    
        # Draw circles on the keypoints
        cv2.circle(output_img, (int(x1),int(y1)), 4, (0, 255, 255), 1)
        cv2.circle(output_img, (int(x2)+c,int(y2)), 4, (0, 255, 255), 1)  
        # Connect the same keypoints
        cv2.line(output_img, (int(x1),int(y1)), (int(x2)+c,int(y2)), (0, 255, 255), 1)
    
    return output_img

all_matches = []
for m, n in matches:
  all_matches.append(m)

img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:30])
# cv2.imwrite('all_matches.png', img3)

# Finding the best matches
good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)

# cv2.imwrite('good_features1.png', cv2.drawKeypoints(img1, [keypoints1[m.queryIdx] for m in good], None, (255, 0, 255)))
# cv2.imwrite('good_features2.png', cv2.drawKeypoints(img2, [keypoints1[m.queryIdx] for m in good], None, (255, 0, 255)))

def warpImages(img1, img2, H, a):

  I = np.float32([[1,0,0],[0,1,0],[0,0,1]])
  A1 = I + a * ( np.linalg.inv(H) - I )
  A2 = H + a * ( I - H )

  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  temp_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
  temp_points_2 = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_1 = cv2.perspectiveTransform(temp_points_1, A1)
  list_of_points_2 = cv2.perspectiveTransform(temp_points_2, A2)

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  
  translation_dist = [-x_min,-y_min]
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(A2), (x_max-x_min, y_max-y_min))
  output_img2 = cv2.warpPerspective(img1, H_translation.dot(A1), (x_max-x_min, y_max-y_min))

  poly1 = np.int32(np.squeeze(cv2.perspectiveTransform(list_of_points_1, H_translation)))
  poly2 = np.int32(np.squeeze(cv2.perspectiveTransform(list_of_points_2, H_translation)))
  output_img_final = process_output_image(output_img, output_img2, poly1, poly2, a)

  return output_img_final

def draw_polygon(ImShape,Polygon,Color):
    Im = np.zeros(ImShape, np.uint8)
    try:
        cv2.fillPoly(Im, [Polygon], Color)
    except:
        try:
            cv2.fillConvexPoly(Im, Polygon, Color)
        except:
            print('cant fill')
    # cv2.fillConvexPoly(Im, Polygon, Color)
 
    return Im

def polygon_overlap(shape, poly1, poly2):
  im1 = draw_polygon(shape, poly1, (122, 122, 122))
  im2 = draw_polygon(shape, poly2, (133, 133, 133))
  im = im1 + im2
  im_sum = np.sum(im, axis=-1)
  overlap_mask = (im_sum == 255 * 3)
  overlap_img = np.zeros_like(im1)
  overlap_img[overlap_mask] = 255
  overlap_img = cv2.cvtColor(overlap_img, cv2.COLOR_RGB2GRAY)
  overlap_contour, hierarchy= cv2.findContours(overlap_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  return overlap_mask, overlap_contour

def process_output_image(output_img, output_img2, poly1, poly2, a):
#   output = np.zeros_like(output_img)
  overlap, contour = polygon_overlap(output_img.shape, poly1, poly2)
  output = output_img + output_img2
  weight_sum = a * output_img + (1 - a) * output_img2
  output[overlap] = weight_sum[overlap]
  contour_img = np.zeros_like(output_img)
  cv2.drawContours(contour_img, contour, -1, (255,255,255), 1)
  contour_img = cv2.cvtColor(contour_img, cv2.COLOR_RGB2GRAY)
  contour_mask = (contour_img == 255)
  contour_img1 = np.zeros_like(output_img)
  contour_img1[contour_mask] = output_img[contour_mask]
  contour_img1_mask = (np.sum(contour_img1, axis=-1) != 0)
  contour_img2 = np.zeros_like(output_img2)
  contour_img2[contour_mask] = output_img2[contour_mask]
  contour_img2_mask = (np.sum(contour_img2, axis=-1) != 0)
  output[contour_img1_mask & ~contour_img2_mask] = output_img[contour_img1_mask & ~contour_img2_mask]
  output[contour_img2_mask & ~contour_img1_mask] = output_img2[contour_img2_mask & ~contour_img1_mask]
  output[contour_img2_mask & contour_img1_mask] = output_img[contour_img2_mask & contour_img1_mask]
  return output

# Set minimum match condition
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    result = warpImages(img2, img1, M, 0.5)

    cv2.imwrite('result.png', result)
