import cv2
import numpy as np
import math

#1. Input path of image file
img1_path = './Foot/left.jpg'
img2_path = './Foot/right.jpg'
img3_path = './Foot/groundtruth.jpg'

Left = cv2.imread(img1_path)
Right = cv2.imread(img2_path)
mid_img = np.zeros((1108,1477,3))
gd = cv2.imread(img3_path)



#2. Set Correspondences of (Right, Left, GroundTruth image )
left_pts = np.array([[229,231],#tl
                     [111,1386],#bl
                     [851,383],#tr
                     [870,1324],#br
                    ])

right_pts = np.array([[257,269], #tl
                      [145,1287], #bl
                      [947,80], #tr
                      [1025,1411],#br
                    ])

gd_pts = np.array([[107,151], #tl
                    [79,1369],#bl
                    [934,105],#tr
                    [995,1360]#br
                   ])

#Will be used for final image
mid_pts = np.array([[0,0],#tl
                    [0,891], #bl
                    [630,0],#tr
                    [630,891],#br
                    ])



#3. Find Homography
cols = 630
rows = 891
H_RM, mask_RM = cv2.findHomography(right_pts, mid_pts,cv2.RANSAC)
H_LM, mask_LM = cv2.findHomography(left_pts, mid_pts,cv2.RANSAC)
H_GM, mask_SM = cv2.findHomography(gd_pts, mid_pts, cv2.RANSAC)


#4.Transform image left and right to center (as middle view)
blank_right = cv2.warpPerspective(Right,H_RM, (cols,rows))
blank_left = cv2.warpPerspective(Left,H_LM, (cols,rows))
blank_gd = cv2.warpPerspective(gd,H_GM, (cols,rows))

# cv2.imshow('Warped Image - Left',blank_left)
# cv2.imshow('Warped Image - Right',blank_right)
# cv2.imshow('Warped Image - Ground Truth',blank_gd)


#convert to Gray for filtering
blank_left_gray = cv2.cvtColor(blank_left, cv2.COLOR_BGR2GRAY)
blank_right_gray = cv2.cvtColor(blank_right,cv2.COLOR_BGR2GRAY)
blank_gd_gray = cv2.cvtColor(blank_gd,cv2.COLOR_BGR2GRAY)

#5.Filter black color
retlL,th1_l = cv2.threshold(blank_left_gray,40,255,cv2.THRESH_BINARY)
retlR,th1_r = cv2.threshold(blank_right_gray,65,255,cv2.THRESH_BINARY)
retlG,th1_g = cv2.threshold(blank_gd_gray,100,255,cv2.THRESH_BINARY)



#remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
th1_l = cv2.morphologyEx(th1_l,cv2.MORPH_OPEN,kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
th1_r = cv2.morphologyEx(th1_r,cv2.MORPH_OPEN,kernel)

# cv2.imshow('Black Fitered Image - Left ',th1_l)
# cv2.imshow('Black Fitered Image - Right ',th1_r)
# cv2.imshow('Black Fitered Image - Ground Truth ',th1_g)




#6. Combine Right and Left foot
combine = blank_left
for i in range(0,rows):
    for j in range(int(cols/2),cols):
        if th1_l[i][j]==0:
            rgb=blank_right[i][j-50]
            r=rgb[0]
            g=rgb[1]
            b=rgb[2]
            combine[i][j]=(r,g,b)



#threshold combine image
combine_gray = cv2.cvtColor(combine, cv2.COLOR_BGR2GRAY)

ret_combine,th_combine = cv2.threshold(combine_gray,40,255,cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
th_combine = cv2.morphologyEx(th_combine,cv2.MORPH_OPEN,kernel)

# cv2.imshow('Combined Left and Right Image', th_combine)


#Detect the black pixel (For calculating and making border)
f_points=[]
for x in range(10,rows-5):
   for y in range(5,cols-5):
      if th_combine[x][y] == 0:
          f_points.append([y,x])
f_points=np.array(f_points)

#convert combine image from gray to BGR, so we can draw colorful line later
th_combine = cv2.cvtColor(th_combine,cv2.COLOR_GRAY2BGR)

#7.Make Border
maxx,maxy=np.argmax(f_points, axis=0)
minx,miny=np.argmin(f_points, axis=0)

#top line
cv2.line(th_combine,(f_points[minx][0],f_points[miny][1]),(f_points[maxx][0],f_points[miny][1]),(0,128,255),2)
cv2.line(combine,(f_points[minx][0],f_points[miny][1]),(f_points[maxx][0],f_points[miny][1]),(0,0,255),2)
# #bottom line
cv2.line(th_combine,(0,f_points[maxy][1]),(f_points[maxx][0],f_points[maxy][1]),(128,128,128),2)
cv2.line(combine,(0,f_points[maxy][1]),(f_points[maxx][0],f_points[maxy][1]),(128,128,128),2)
#left line
cv2.line(th_combine,(f_points[minx][0],f_points[miny][1]),(f_points[minx][0],f_points[maxy][1]),(0,128,255),2)
cv2.line(combine,(f_points[minx][0],f_points[miny][1]),(f_points[minx][0],f_points[maxy][1]),(0,0,255),2)
#right line
cv2.line(th_combine,(f_points[maxx][0],0),(f_points[maxx][0],f_points[maxy][1]),(128,128,128),2)
cv2.line(combine,(f_points[maxx][0],0),(f_points[maxx][0],f_points[maxy][1]),(128,128,128),2)


#8. Create measurement of foot based on length and width of border
text_length=str(round((f_points[maxy][1]-f_points[miny][1])/3,1))
text_width=str(round((f_points[maxx][0]-f_points[minx][0])/3,1))
cv2.putText(th_combine, text_length+'mm', (f_points[minx][0]-50, int((f_points[maxy][1]-f_points[miny][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (73,83,255), 1, cv2.LINE_AA)
cv2.putText(th_combine, text_width+'mm', (int((f_points[maxx][0] - f_points[minx][0])/2)+f_points[minx][0],f_points[miny][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (73,83,255), 1, cv2.LINE_AA)
cv2.putText(th_combine, 'Size: ('+text_length+' x '+ text_width+' mm)', (int(cols/2),rows-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA)

cv2.putText(combine, text_length+'mm', (f_points[minx][0]-50, int((f_points[maxy][1]-f_points[miny][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (73,83,255), 1, cv2.LINE_AA)
cv2.putText(combine, text_width+'mm', (int((f_points[maxx][0] - f_points[minx][0])/2)+f_points[minx][0],f_points[miny][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (73,83,255), 1, cv2.LINE_AA)


#9. Creating Ruler
w = f_points[maxx][0] #X axis
for i in range(int(math.ceil((f_points[maxx][0]-f_points[minx][0])/3)),0,-1):
    if(i%50==0):
        cv2.putText(th_combine, str(i), ((w - (i * 3), 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (128, 128, 128), 1, cv2.LINE_AA)
        cv2.line(th_combine, (w - (i * 3), 10), (w - (i * 3), 1), (128, 128, 128), 1)
    elif(i%10==0):
        cv2.line(th_combine, (w - (i * 3), 5), (w - (i * 3), 1), (128, 128, 128), 1)


d = f_points[maxy][1]
for j in range (0, int((f_points[maxy][1]-f_points[miny][1])/3), 1):
    if (j % 50 == 0):
        cv2.putText(th_combine, str(j), (f_points[maxx][0]+11, (j*3)+f_points[miny][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (128, 128, 128), 1,cv2.LINE_AA)
        cv2.line(th_combine, (f_points[maxx][0], (j*3)+f_points[miny][1]), (f_points[maxx][0]+10, (j*3)+f_points[miny][1]), (128, 128, 128), 1)
    elif (j % 10 == 0):
        cv2.line(th_combine, (f_points[maxx][0], (j*3)+f_points[miny][1]), (f_points[maxx][0]+5, (j*3)+f_points[miny][1]), (128, 128, 128), 1)

# cv2.imshow('Measured Foot Print - Estimated', th_combine)

#NOW LETS DEAL WITH Ground Truth (Same way)
gd_points = []
for i in range(10,rows-10):
    for j in range(10,cols-10):
        if th1_g[i][j]==0:
            gd_points.append([j,i])
gd_points=np.array(gd_points)

maxsx,maxsy=np.argmax(gd_points, axis=0)
minsx,minsy=np.argmin(gd_points, axis=0)

#top line
cv2.line(blank_gd,(gd_points[minsx][0],gd_points[minsy][1]),(gd_points[maxsx][0],gd_points[minsy][1]),(0,128,255),2)
# #bottom line
cv2.line(blank_gd,(0,gd_points[maxsy][1]),(gd_points[maxsx][0],gd_points[maxsy][1]),(255,0,0),2)

#left line
cv2.line(blank_gd,(gd_points[minsx][0],gd_points[minsy][1]),(gd_points[minsx][0],gd_points[maxsy][1]),(0,128,255),2)

#right line
cv2.line(blank_gd,(gd_points[maxsx][0],0),(gd_points[maxsx][0],gd_points[maxsy][1]),(255,0,0),2)



#10. Give measurement based on length and width of border
text_length = str(round((gd_points[maxsy][1]-gd_points[minsy][1])/3,1))
text_width = str(round((gd_points[maxsx][0]-gd_points[minsx][0])/3,1))
cv2.putText(blank_gd, text_length+'mm', (gd_points[minsx][0]-70, gd_points[minsy][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
cv2.putText(blank_gd, text_width+'mm', (int((gd_points[maxsx][0] - gd_points[minsx][0])/2)+gd_points[minsx][0],gd_points[minsy][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
cv2.putText(blank_gd, 'Size: ('+text_length+' x '+ text_width+' mm)', (int(cols/2),rows-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)

# cv2.imshow('Measured Foot Print - Ground Truth', blank_gd)

#11. Compare Estimated and Ground Truth
combine = np.concatenate((th_combine, blank_gd), axis=1)
cv2.imshow("Estimated vs Ground Truth", combine)




cv2.waitKey()