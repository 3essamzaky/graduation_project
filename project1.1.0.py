import cv2 
import numpy as np
import mediapipe as mp
import math
#import serial
import time

mp_face_mesh= mp.solutions.face_mesh
TOP_EYE=[159]
BOTTOM_EYE=[145]

LEFT_EYE=[362,382,381,380,374,373,390,249,263,466,387,386,385,384,398]
RIGHT_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

L_H_LEFT=[33]
L_H_RIGHT=[133]
R_H_LEFT=[362]
R_H_RIGHT=[263]
RIGHT_IRIS=[474,475,476, 477]
LEFT_IRIS=[469,470,471, 472]

#if _name_ == 'main':
    #ser = serial.Serial('/dev/ttyACM0',9600, timeout=0.1)
    
def euclidean_distance(point1,point2):
    x1,y1=point1.ravel()
    x2,y2=point2.ravel()
    distance=math.sqrt((x2-x1)**2 +(y2-y1)**2)
    

    return distance

def iris_position(iris_center,right_point,left_point,TOP_EYE,BOTTOM_EYE):
    center_to_right_dist=euclidean_distance(iris_center,right_point)
    center_to_left_dist=euclidean_distance(iris_center,left_point)
    TOP_BOTTOM=euclidean_distance(TOP_EYE,BOTTOM_EYE)
    total_dist=euclidean_distance(right_point,left_point)
    ratio=center_to_right_dist/total_dist
    iris_position=""

    
    if ratio <=0.42 and TOP_BOTTOM>5:
        iris_position="2"
        #ser.write(iris_position.encode('utf-8'))
        #time.sleep(0.03)
        
    elif ratio>0.42 and ratio<=0.5 and TOP_BOTTOM>5:
        
        iris_position="1"
        #ser.write(iris_position.encode('utf-8'))
        #time.sleep(0.03)

        
    elif ratio>0.5 and TOP_BOTTOM>5:
        iris_position="3"
        #ser.write(iris_position.encode('utf-8'))
        #time.sleep(0.03)
   
    
    else:
          
          iris_position="eye close"
          #time.sleep(0.3)

     
    
    #print(TOP_BOTTOM)
        
    return iris_position,ratio


#def eye_close(TOP_EYE,BOTTOM_EYE):
      #TOP_BOTTOM=euclidean_distance(TOP_EYE,BOTTOM_EYE)
      #print(TOP_BOTTOM)
      
      #if TOP_BOTTOM<=5:
          #print("eye close")
     

      
   
cap=cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
    while True:
        
        ret,frame=cap.read()
        
        if not ret:
            break
        frame=cv2.flip(frame,1)
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_h,img_w=frame.shape[:2]
        results=face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks :
            
            mesh_points=np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int)for p in results.multi_face_landmarks[0].landmark])
            (l_cx,l_cy),l_radius=cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy),l_radius=cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            (t_cx,t_cy),l_radius=cv2.minEnclosingCircle(mesh_points[TOP_EYE])
            (b_cx,b_cy),l_radius=cv2.minEnclosingCircle(mesh_points[BOTTOM_EYE])
            center_left=np.array([l_cx,l_cy],dtype=np.int32)
            center_right=np.array([r_cx,r_cy],dtype=np.int32)
            TOP=np.array([t_cx,t_cy],dtype=np.int32)
            BOTTOM=np.array([b_cx,b_cy],dtype=np.int32)
            cv2.circle(frame,center_left,int(l_radius),(255,0,255),1,cv2.LINE_AA)
            cv2.circle(frame,center_right,int(l_radius),(255,0,255),1,cv2.LINE_AA)
            iris_pos,ratio=iris_position(center_right,mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0],TOP,BOTTOM)
            print(iris_pos)
            #eye_close(TOP,BOTTOM)
           
        else:
            noeye="4"
            #ser.write(noeye.encode('utf-8') )
            print(noeye)

        cv2.imshow('img',frame)
        key=cv2.waitKey(30)
        if key==ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()