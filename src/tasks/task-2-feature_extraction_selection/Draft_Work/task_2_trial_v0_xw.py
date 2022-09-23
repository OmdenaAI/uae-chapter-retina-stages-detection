#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


img1= cv2.imread('1.jpeg')


# In[3]:


plt.imshow(img1, cmap='gray')


# In[4]:


# apply gray scale filter
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


# In[10]:


# Width and heigth the image
height, width= gray.shape


# In[11]:


# Sum the value lines 
vertical_px = np.sum(gray, axis=0)


# In[12]:


normalize=vertical_px/255


# In[13]:


# create a black image with zeros 
blankImage = np.zeros_like(gray)


# In[14]:


# Make the vertical projection histogram
for idx, value in enumerate(normalize):
    cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)


# In[15]:


plt.imshow(blankImage,cmap='gray')


# In[16]:


edges = cv2.Canny(blankImage,100,200)


# In[33]:


# https://stackoverflow.com/questions/32146633
# http://stackoverflow.com/a/29799815/1698058
# Get index of matching value.
#@jit(nopython=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

bounds = [50, 150]
# Now the points we want are the lowest-index 255 in each row
window = edges[bounds[1]:bounds[0]:-1].transpose()

xy = []
for i in range(len(window)):
    col = window[i]
    j = find_first(255, col)
    if j != -1:
        xy.extend((i, j))
# Reshape into [[x1, y1],...]
data = np.array(xy).reshape((-1, 2))
# Translate points back to original positions.
data[:, 1] = bounds[1] - data[:, 1]


# In[34]:


xdata = data[:,0]
ydata = data[:,1]

z = np.polyfit(xdata, ydata, 10)
f = np.poly1d(z)


# In[35]:


t = np.arange(0, edges.shape[1], 1)
plt.figure(2, figsize=(8, 16))
ax1 = plt.subplot(211)
ax1.imshow(edges,cmap = 'gray')
ax2 = plt.subplot(212)
ax2.axis([0, edges.shape[1], edges.shape[0], 0])
ax2.plot(t, f(t))
plt.show()


# In[ ]:




