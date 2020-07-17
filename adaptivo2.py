import cv2
import numpy as np
import os


def Thresholding(image1):
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	res = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	h, w = img1.shape
	winsize=14  #tamaño de la subventana para cada pixel
	const=6 #constate


	for i in range(h):
	    for j in range(w):
	    	y0=i-int(winsize/2)
	    	y1=i+int(winsize/2)+1
	    	x0=j-int(winsize/2)
	    	x1=j+int(winsize/2)+1

	    	if y0 < 0:
	    		y0 = 0
	    	if y1 > h: #filas
	    		y1 = h
	    	if x0 < 0:
	    		x0 = 0
	    	if x1 > w:	#columnas
	    		x1 = w
	    	
	    	block =img1[y0:y1, x0:x1]
	    	#print(block)

	    	threshold =np.mean(block)-const # sacamos el promedio de la sub ventana restado por la constante
	    	if img1[i,j]<threshold:
	    		res[i,j] = 0
	    	else:
	    		res[i,j] = 255

	imgout = res
    # img = cv2.imwrite("prueba.png", res)
 	
# 	kernel = np.ones((3,3), np.uint8)       # kernel como matrix 3x3 de
#     #Crear imagen de erosión y dilatación a partir de la imagen original
# 	dilate_image = cv2.dilate(imgout, kernel, iterations=1)
# 	resultado = cv2.erode(dilate_image, kernel, iterations=1)
    
 	#################3
	
	kernel2 = np.ones((3,3), np.uint8)
    kernel1 = np.ones((2,2), np.uint8)

    # a = cv2.erode(imgout , kernel1, iterations=2)
	a1  = cv2.dilate(imgout, kernel1, iterations=1)
	a2 = cv2.erode(a1, kernel1, iterations=1)
	resultado = cv2.GaussianBlur(a2,(3,3),0)

    ##################
	return resultado
# 	return res


imagen1 = cv2.imread("prueba2.png")
result = Thresholding(imagen1)
cv2.imwrite("prueba2-2-2-1.png", result)
cv2.waitKey(0)  


 