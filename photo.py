import cv2
import numpy as np

def dummy(val):
	pass

color_original = cv2.imread('test.jpg')
color_modified = cv2.imread('test.jpg')

gray_original = cv2.cvtColor(color_original,cv2.COLOR_BGR2GRAY)
gray_modified = cv2.cvtColor(color_original,cv2.COLOR_BGR2GRAY)

identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gaussian_kernel1 = cv2.getGaussianKernel(3,0)
gaussian_kernel2 = cv2.getGaussianKernel(5,0)
box_kernel = np.array([1,1,1],[1,1,1],[1,1,1],np.float32)/9

kernels = [identity_kernel,sharpen_kernel,gaussian_kernel1,gaussian_kernel2,box_kernel]

cv2.namedWindow('Photo Editing')
cv2.createTrackbar('Brightness','Photo Editing',1,100,dummy)
cv2.createTrackbar('Contrast','Photo Editing',50,100,dummy)
cv2.createTrackbar('Filters','Photo Editing',0,1,dummy)
cv2.createTrackbar('Grayscale','Photo Editing',0,1,dummy)

while True:
	grayscale = cv2,getTrackbarPos('Grayscale','Photo Editing')
	if grayscale== 0:
		cv2.imshow('Photo Editing',color_modified)
	else:
		cv2.imshow('Photo Editing',gray_modified)
	
	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
		
	contrast = cv2.getTrackbarPos('Contrast','Photo Editing')
	brightness = cv2.getTrackbarPos('Brightness','Photo Editing')
	kernel = cv2.getTrackbarPos('Filter','Photo Editing')
	
	color_modified = cv2.filter2D(color_original,-1,kernels[kernel])
	gray_modified = cv2.filter2D(gray_original,-1,kernels[kernel])

	color_modified = cv2.addWeighted(color_original,contrast,np.zeros(color_original.shape,dtype=color_original.dtype),0,brightness-50)
	gray_modified = cv2.addWeighted(gray_original,contrast,np.zeros(gray_original.shape,dtype=color_original.dtype),0,brightness-50)
	
	cv2.destroyAllWindows()
