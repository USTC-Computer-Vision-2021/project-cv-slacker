import cv2
import numpy as np

# input video
# parameter: t0,[t1],t2,t3
# output: new video

video1 = cv2.VideoCapture("01.mp4")
video2 = cv2.VideoCapture("02.mp4")

t1 = 3.3 # unit: second
overallcontrol = 5

fps_Origin = video1.get(cv2.CAP_PROP_FPS) # 原视频的帧率
fps = fps_Origin # 需要保存视频的帧率
# 低分辨率图像的长和宽
width_origin = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
width = width_origin//overallcontrol
height_origin = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
height = height_origin//overallcontrol


# 各个时间节点对应的帧数
num1 = int(t1 * fps) # t1
num2 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT)) # t2
deltanum = num2 - num1 - 1 # delta t

# 设置视频读取位置
video1.set(cv2.CAP_PROP_POS_FRAMES, num1)
video2.set(cv2.CAP_PROP_POS_FRAMES, 0)

imgs10 = []
imgs20 = []
imgs1 = []
imgs2 = []

# 读取视频各帧
for i in range(deltanum):
  _, frame = video1.read()
  imgs10.append(frame)
  _, frame = video2.read()
  imgs20.append(frame)
  
  imgs1.append(cv2.resize(imgs10[i], (int(width), int(height))))
  imgs2.append(cv2.resize(imgs20[i], (int(width), int(height))))
    
def generateHMatrix(imgs1, imgs2):
  H_Array = []
  for i in range(len(imgs1)):
    img1 = imgs1[i]
    img2 = imgs2[i]
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.SIFT_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher_create(cv2.NORM_L1)

    matches = bf.knnMatch(descriptors1, descriptors2,k=2)
    all_matches = []
    for m, n in matches:
      all_matches.append(m)
    
    good = []
    for m, n in matches:
      if m.distance < 0.6 * n.distance:
        good.append(m)

    MIN_MATCH_COUNT = 5

    if len(good) > MIN_MATCH_COUNT:
      src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
      dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

      M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
      H_Array.append(M)
  return H_Array

def smoothHMatrix(HMatrixArray):
  # TODO
  return HMatrixArray

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

  # TODO: optimize this
  output[contour_img1_mask & ~contour_img2_mask] = output_img[contour_img1_mask & ~contour_img2_mask]
  output[contour_img2_mask & ~contour_img1_mask] = output_img2[contour_img2_mask & ~contour_img1_mask]
  output[contour_img2_mask & contour_img1_mask] = output_img[contour_img2_mask & contour_img1_mask]
  return output

def getCenters(H_array):
  length = len(H_array)
  a = 0
  centers = []
  output_heights = []
  for i in range(length):
    H = H_array[i]
    a = i/length
    I = np.float32([[1,0,0],[0,1,0],[0,0,1]])
    A1 = I + a * ( np.linalg.inv(H) - I )
    A2 = H + a * ( I - H )
    temp_point = np.float32([[[ height/2, width/2]]])
    center1 = cv2.perspectiveTransform(temp_point, A1)
    center2 = cv2.perspectiveTransform(temp_point, A2)
    center = (1-a)*center1+(a)*center2
    centers.append(center)
    mask1 = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
    mask2 = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
    mask0 = warpImages(mask1, mask2, H, a)
    _, mask = cv2.threshold(mask0.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    output_height=2
    while 1: 
      output_height = output_height + 2
      success, temp = cutimg(mask, np.squeeze(centers[i]), output_height)
      if (not success) or (np.min(temp) == 0):
        break
    output_heights.append(output_height)
  return centers, output_heights

def cutimg(inputimg, center0, height0):
  width0 = height0 / height * width
  if(int(center0[0]-height0/2) < 0 or int(center0[1]-width0/2) < 0):
    return 0, -1
  if(int(center0[0]+height0/2) > inputimg.shape[0] or int(center0[1]+width0/2) > inputimg.shape[1]):
    return 0, -1
  return 1, inputimg[int(center0[0]-height0/2) : int(center0[0]+height0/2), int(center0[1]-width0/2) : int(center0[1]+width0/2)]

def smoothHeights(heights):
  # TODO
  return heights

H_array0 = generateHMatrix(imgs1, imgs2)

H_array_final = smoothHMatrix(H_array0)

centers = []
new_heights = []
centers, new_heights = getCenters(H_array_final)

# 需要处理一下new_heights以平滑过渡
final_heights = smoothHeights(new_heights)

outputimgs = []
for i in range(deltanum):
  img1 = imgs10[i]
  img2 = imgs20[i]
  temp = warpImages(img1, img2, H_array_final[i], i/deltanum)
  testtest = final_heights[i]*overallcontrol
  outputimgs.append(cutimg(temp, np.squeeze(centers[i]*overallcontrol),final_heights[i]*overallcontrol)[1])


videoWriter =cv2.VideoWriter('videoOut.avi', cv2.VideoWriter_fourcc('X','V','I','D'), fps, (int(width_origin), int(height_origin)))
video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(num1):
  _, frame = video1.read()
  videoWriter.write(frame)
for i in range(deltanum):
  frame = cv2.resize(outputimgs[i],(width_origin, height_origin))
  videoWriter.write(frame)
video2.set(cv2.CAP_PROP_POS_FRAMES, deltanum)
while(1):
  test, frame = video2.read()
  if(test):
    videoWriter.write(frame)
  else:
    break


video1.release()
video2.release()
videoWriter.release()