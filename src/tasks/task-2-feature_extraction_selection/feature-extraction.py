import numpy as np
import cv2

img = cv2.imread('./DME train/11.jpeg')
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
X_data = []
Y_data = []

x,y,c = bw_img.shape


for Y in range(y):
   for X in range(x):
       if bw_img[X][Y][0] == 255:
            X_data.append(Y)
            Y_data.append(x - X)
            break


features = np.polyfit(X_data, Y_data, 10)
