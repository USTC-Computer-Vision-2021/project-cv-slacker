import sys
import cv2
import numpy as np

# input vedio
# parameter: t0,[t1],t2,t3
# output: new VisibleDeprecationWarning




# steps:
# 1.cut vedio1 and vedio2 , 提取两组帧：
# frame10
# frame20
# 2.according each frame, generate frames with更小的分辨率
# frame1
# frame2
# 3.zccording frame1 and frame2, generate H
# 4.do something to H
# 5.put H to frame10 and frame20, derive frame_output

# 后续：
# 不同帧率不同大小之类的是否也可以拼一拼？
# 如果有场景切换比较大的，要如何进行拼图？
# 整体的光场如何调整？
# 关键帧提取以提高处理速度
# 声音？

vedio1 = cv2.VideoCapture("test1.mp4")
vedio2 = cv2.VideoCapture("test2.mp4")

t1 = 7 # unit: second
overallcontrol = 3

fps_Origin = vedio1.get(cv2.CAP_PROP_FPS) # 原视频的帧率
fps = fps_Origin # 需要保存视频的帧率
# size_Origin = (vedio1.get(cv2.CAP_PROP_FRAME_WIDTH), vedio1.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 原视频的大小
# size = size_Origin # 需要保存视频的大小

width = vedio1.get(cv2.CAP_PROP_FRAME_WIDTH) // overallcontrol
height = vedio1.get(cv2.CAP_PROP_FRAME_HEIGHT) // overallcontrol
videoWriter =cv2.VideoWriter('videoOut.avi', cv2.VideoWriter_fourcc('X','V','I','D'), fps, (int(width), int(height)))



num1 = int(t1 * fps)
num2 = int(vedio1.get(cv2.CAP_PROP_FRAME_COUNT))
deltanum = num2 - num1 + 1

vedio1.set(cv2.CAP_PROP_POS_FRAMES, num1)
vedio2.set(cv2.CAP_PROP_POS_FRAMES, 0)

imgs10 = []
imgs20 = []
imgs1 = []
imgs2 = []

for i in range(0,deltanum-1):
    _, frame = vedio1.read()
    imgs10.append(frame)
    _, frame = vedio2.read()
    imgs20.append(frame)
    
    imgs1.append(cv2.resize(imgs10[i], (width, height)))
    imgs2.append(cv2.resize(imgs20[i], (width, height)))
    # videoWriter.write(frame)
    
def generateHMatrix(imgs1, imgs2):
    H_Arrey = []
    for i in range(len(imgs1)):
        img1 = imgs1[i]
        img2 = imgs2[i]
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Create our ORB detector and detect keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=2000)
        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
        # Create a BFMatcher object.
        # It will find all of the matching keypoints on two images
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        # Find matching points
        matches = bf.knnMatch(descriptors1, descriptors2,k=2)
        all_matches = []
        for m, n in matches:
            all_matches.append(m)
        # Finding the best matches
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)

            # Set minimum match condition
        MIN_MATCH_COUNT = 5

        if len(good) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography
            src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            # Establish a homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            H_Arrey.append(M)
    return H_Arrey

def smoothHMatrix(HMatrixArrey):
    return HMatrixArrey

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
    return overlap_mask

def process_output_image(output_img, output_img2, poly1, poly2, a):
#   output = np.zeros_like(output_img)
    overlap = polygon_overlap(output_img.shape, poly1, poly2)
    output = output_img + output_img2
    weight_sum = a * output_img + (1 - a) * output_img2
    output[overlap] = weight_sum[overlap]
    return output

def getCenters(H_arrey):
    return H_arrey, H_arrey
    
H_arrey0 = generateHMatrix(imgs1, imgs2)
H_arrey_final = smoothHMatrix(H_arrey0)

centers = []
zooms = []
centers, zooms = getCenters(H_arrey_final)

outputimgs = []
for i in range(0, deltanum):
    img1 = imgs10[i]
    img2 = imgs20[i]
    outputimgs.append(warpImages(img1, img2, H_arrey_final[i], i/deltanum))

# final step: write to file
vedio1.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(0, num1):
    _, frame = vedio1.read()
    videoWriter.write(frame)
for i in range(0, deltanum):
    frame = outputimgs[i]
    videoWriter.write(frame)
vedio2.set(cv2.CAP_PROP_POS_FRAMES, deltanum)
for i in range(0, num1):
    test, frame = vedio2.read()
    if(test):
        videoWriter.write(frame)
    else:
        break


vedio1.release()
vedio2.release()
videoWriter.release()