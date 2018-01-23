def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

def face_landmarks(image, detector, predictor):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor

    # load the input image, resize it, and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        #determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
 
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        #(x, y, w, h) = rect_to_bb(rect)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
        # show the face number
        #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        #for num,(x, y) in enumerate(shape):
        #    if num == 30 or num == 8 or num == 45 or num == 36 or num == 54 or num == 48 : cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    return shape
 
def draw_game(image):
    
    center = (int(width/2),int(height/2))
    length = x ** 2 + y ** 2
    x = x * 10 / length
    y = y * 10 / length
    target_x = center[0] + x
    target_y = center[1] + y
    target = (target_x, target_y)
    
    length = x ** 2 + y ** 2
    
    image = cv2.circle(image,center,10,(0,0,255), -1)
    
    cv2.line(image, center, target, (255,0,0), 2)
    
    return image
    
# import the necessary packages
import numpy as np
import dlib
import cv2
import time
import random

debug_show = True
scale = 2

mode = 0
game_time = 0.0
start_time = time.time()

bullets = []
speeds = []
enemies = []


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

if (not cap.isOpened()):
    print ("Camera not opened!")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if (ret == False):
        print("Camera read failed!")
        break;
       
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    
    height = int(height / scale)
    width = int(width / scale)
    
    
    if (not debug_show):
        image = np.zeros((height,width,3), np.uint8)
    else:
        image = cv2.resize(frame,(int(width), int(height)), interpolation = cv2.INTER_CUBIC)

    frame = cv2.resize(frame,(int(width), int(height)), interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        #determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
    
        for num,(x, y) in enumerate(shape):
            if num == 30 or num == 8 or num == 45 or num == 36 or num == 54 or num == 48 :
                if debug_show :
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    size = image.shape

    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                            shape[30],     # Nose tip 31
                            shape[8],     # Chin 9
                            shape[45],     # Left eye left corner 46
                            shape[36],     # Right eye right corne 37
                            shape[54],     # Left Mouth corner 55
                            shape[48]      # Right mouth corner 49
                        ], dtype="double")

    # 3D model points.
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        
                        ])


    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    #print ("Camera Matrix :\n {0}".format(camera_matrix));

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    #print ("Rotation Vector:\n {0}".format(rotation_vector))
    #print ("Translation Vector:\n {0}".format(translation_vector))


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose


    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    if (debug_show):
        for p in image_points:
            cv2.circle(image, (int(p[0]), int(p[1])), 5, (0,0,255), -1)
    
        cv2.line(image, p1, p2, (255,0,0), 2)
    
    ####GAME PART####
    
    # move the turret
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    center = (int(width/2),int(height))
    length = (x ** 2 + y ** 2) ** 0.5
    x = x * 20 / length
    y = y * 20 / length
    target_x = center[0] + x
    target_y = center[1] + y
    target = (int(target_x), int(target_y))
    speed = (int (x / 5),int(y / 5))
    
    
    # move bullets
    for i in range(len(bullets)):
        new_pos = (bullets[i][0] + speeds[i][0], bullets[i][1] + speeds[i][1])
        del bullets[i]
        bullets.insert(i,new_pos)
        
    # move enemies
    for i in range(len(enemies)):
        new_pos = (enemies[i][0], enemies[i][1] + 1 + int(0.01 * game_time))
        del enemies[i]
        enemies.insert(i,new_pos)
        if (new_pos[1] > height):
            mode = 3
        
    # garbage collection
    for i in range(len(bullets)):
        pos = (bullets[i][0], bullets[i][1])
        if ((bullets[i][1] < 0) or (bullets[i][1] > height) or (bullets[i][0] < 0) or (bullets[i][0] > width)):
            bullets.remove(pos)
            del speeds[i]
            break;
    
    for i in range(len(enemies)):
        pos = (enemies[i][0], enemies[i][1])
        if (enemies[i][1] > height):
            enemies.remove(pos)
            break;
        
    # collision detection
    
    hit_bullet = -1
    hit_enemy = -1
    
    for i in range(len(enemies)):
        enemy_pos = (enemies[i][0], enemies[i][1])
        for j in range(len(bullets)):
            bullet_pos = (bullets[j][0], bullets[j][1])
            if ((abs(bullet_pos[0] - enemy_pos[0]) < 5) and (abs(bullet_pos[1] - enemy_pos[1]) < 5)):
                hit_bullet = j
                hit_enemy = i
                break;
              
    if (hit_bullet != -1):
        del bullets[hit_bullet]
        del speeds[hit_bullet]
        del enemies[hit_enemy]
    
    # update game time
    if (mode == 1):
        game_time = 0.0
        bullets = []
        speeds = []
        enemies = []
        last_bullet = 0.0
        last_spawn = 0.0
        start_time = time.time()
        mode = 2;
    elif (mode == 2):
        game_time = time.time() - start_time
        # spawn bullet
        if (game_time - last_bullet >= 0.2):
            bullets.append(target)
            speeds.append(speed)
            last_bullet = game_time
        # spawn enemy
        if (game_time - last_spawn >= 1.0):
            enemy=(random.randint(10,width-10),0)
            enemies.append(enemy)
            last_spawn = game_time
    elif (mode == 3):
        game_time = game_time
        
    
    # draw turret
    if (mode != 0):
        cv2.circle(image,center,10,(0,0,255), -1)
        cv2.line(image, center, target, (255,0,0), 2)
    
    # draw bullets
    for pos in bullets:
        cv2.circle(image,pos,2,(255,255,255), -1)
        
    # draw enemies
    for pos in enemies:
        cv2.circle(image,pos,6,(0,255,0), -1)
    
    # draw text
    if (mode == 0):
        cv2.putText(image,'Press S to start', (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    elif (mode == 2):
        cv2.putText(image,'Time : ' + str(int(game_time)), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    elif (mode == 3):
        cv2.putText(image,'Time : ' + str(int(game_time)), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,'Game Over!', (0,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    ####################

    # Display the resulting frame
    cv2.imshow('frame',image)
    k = cv2.waitKey(1)
    if (k == ord('q')):
        break
    elif (k == ord('d')):
        debug_show = not debug_show
    elif (k == ord('s')):
        mode = 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()