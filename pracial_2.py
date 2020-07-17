import cv2
import numpy as np
import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import copy
from operator import itemgetter


class Aplicacion:
    def __init__(self):
        self.ventana1 = Tk()
        self.ventana1.title("evaluacion 2")
        self.img = PhotoImage()
        self.abc = cv2.cvtColor
        self.revisor = bool
        self.revisor = 0
        self.errores = DoubleVar()
        self.etiqueta = DoubleVar()
        self.auto = bool
        self.d = DoubleVar(value=1)

        self.canvas1 = Canvas(self.ventana1, width=1200, height=600, background="black")

        self.vector_x = []
        self.vector_y = []
        self.canvas1.bind("<Motion>", self.mover_mouse)
        self.canvas1.bind("<Button-1>", self.presion_mouse)
        self.canvas1.grid(column=0, row=1)
        self.boton2 = ttk.Button(self.ventana1, text="Subir Imagen",command=self.open_img)
        self.boton2.place(x=450, y=20, width=120, height=50)
        self.boton3 = ttk.Button(self.ventana1, text="Procesar Imagen",command=self.open_img_2)
        self.boton3.place(x=600, y=20, width=120, height=50)

        self.etiq2 = ttk.Label(self.ventana1, textvariable=self.etiqueta,
                               foreground="yellow", background="black",
                               borderwidth=5, anchor="e")
        self.etiq2.place(x=450, y=80, width=150, height=20)
        self.etiqueta.set("     Imagen de salida ")

        self.etiq1 = ttk.Label(self.ventana1, textvariable=self.errores,
                               foreground="yellow", background="black",
                               borderwidth=5, anchor="e") 
        self.etiq1.place(x=450, y=120, width=250, height=40)
        
        self.boton_salir = ttk.Button(self.ventana1, text="Salir", 
                                 command=quit)
        self.boton_salir.place(x=500, y=520, width=150, height=40)

        self.etiq = ttk.Label(self.ventana1, 
                               text="Imagen de salida: ")

        self.combo = ttk.Combobox(self.ventana1,values=["colores", "escala_de_grices","blanco_y_negro"])
        self.combo.place(x=600, y=80, width=100, height=20)

        self.boton4 = ttk.Button(self.ventana1, text= "iluminacion", command = self.multiplicacion)
        self.boton4.place(x=450, y=150, width = 100, height = 20)

        self.c_d = ttk.Entry(self.ventana1, textvariable=self.d, width=10)
        self.c_d.place(x=580, y=150, width=50, height=20)

        self.ventana1.mainloop()

    def presion_mouse(self, evento):
        if(evento.x>50 and evento.x<430 and evento.y >50 and evento.y < 550):
            if(self.revisor == 0):
                self.errores.set("¡ERROR! no hay imagen seleccionada")
            else:
                if(len(self.vector_x) <= 3 ):
                    ca = self.canvas1.create_oval(evento.x-3,evento.y-3,evento.x+3,evento.y+3, fill="blue")
                    temp_x = (evento.x - 50) 
                    temp_y = (evento.y - 50) 
                    self.vector_x.append(temp_x)
                    self.vector_y.append(temp_y)
                else:
                    self.errores.set("¡ERROR! ya agregaste 4 puntos")

    def mover_mouse(self, evento):        
        self.ventana1.title(str(evento.x)+" - "+str(evento.y))

    def open_img(self, *args):
        self.vector_x = []
        self.vector_y = []
        filename = filedialog.askopenfilename(title='open') 
        x = filename
        self.img = x
        img_o = cv2.imread(x)
        self.abc = cv2.cvtColor(img_o,1)
        img = Image.open(filename)  # PIL solution
        img = img.resize((380, 500), Image.ANTIALIAS) #The (250, 250) is (height, width)
        self.photo = ImageTk.PhotoImage(img) #llamar imagen 
        self.canvas1.create_image(50, 50, image=self.photo, anchor=NW)#abrir en ventana
        self.errores.set("Imagen lista para analizar")
        self.revisor = 1
        return

    def open_img_2(self, *args):
        #foto =  self.Thresholding()
        if(self.revisor == 0):
                self.errores.set("¡ERROR! no hay imagen seleccionada")
        else:
            val = self.combo.get()

            if(val == 'colores' or val == 'escala_de_grices' or val == 'blanco_y_negro'  ):
                image1 = self.abc
                x,y,z = image1.shape
            else:
                self.errores.set("¡ERROR! no hay opcion seleccionada ")
                return
            if(len(self.vector_x) == 4):
                self.auto = 0
                x_2 = x/500
                y_2 = y/380
                pts1 = np.float32([[(self.vector_x[0])*x_2,self.vector_y[0]*y_2],
                                    [(self.vector_x[1])*x_2,self.vector_y[1]*y_2],
                                    [(self.vector_x[2])*x_2,self.vector_y[2]*y_2],
                                    [(self.vector_x[3])*x_2,self.vector_y[3]*y_2]])
                print(pts1)
                pts2 = np.float32([[0,0],[x,0],[x,y],[0,y]])

                if (val == 'colores'):
                    m = self.getPerspectiveTransform(pts1,pts2)
                    foto = cv2.warpPerspective(image1,m,(x,y))
                    foto = self.procesar_colores(foto)
                    print('colores')

                elif (val == 'escala_de_grices' ):
                    self.errores.set(" Procesando imagen ")
                    m = self.getPerspectiveTransform(pts1,pts2)
                    foto = cv2.warpPerspective(image1,m,(x,y))
                    foto = self.gray(foto)
                elif (val == 'blanco_y_negro' ):
                    m = self.getPerspectiveTransform(pts1,pts2)
                    foto = cv2.warpPerspective(image1,m,(x,y))
                    foto = self.Thresholding(foto)
                    self.errores.set(" Procesando imagen ")
                    print('byn')

                else:
                    self.errores.set(" no opcion seleccionada! ")
                
            else:
                #pts1 = funcion()
                self.errores.set(" Procesando imagen ")
                pts2 = np.float32([[0,0],[x,0],[x,y],[0,y]])
                x___ = self.harris_corners(image1)
                pts1 = np.float32([[x___[2][0],x___[2][1]],
                                    [x___[3][0],x___[3][1]],
                                    [x___[0][0],x___[0][1]],
                                    [x___[1][0],x___[1][1]]])
                self.auto = 1
                print(pts1)
                if (val == 'colores'):
                    m = self.getPerspectiveTransform(pts1,pts2)
                    foto = cv2.warpPerspective(image1,m,(x,y))
                    foto = self.procesar_colores(foto)
                    print('colores')

                elif (val == 'escala_de_grices' ):
                    m = self.getPerspectiveTransform(pts1,pts2)
                    foto = cv2.warpPerspective(image1,m,(x,y))
                    foto = self.gray(foto)
                    
                elif (val == 'blanco_y_negro' ):
                    m = self.getPerspectiveTransform(pts1,pts2)
                    foto = cv2.warpPerspective(image1,m,(x,y))
                    foto = self.Thresholding(foto)
                    print('byn')
                else:
                    self.errores.set(" no opcion seleccionada! ")

            cv2.imwrite("add.png", foto)
            img = Image.open("add.png")
            if(self.auto):
                img_2 = Image.open("esquinas.png")
            else:
                img_2 = Image.open(self.img)

            img = img.resize((380, 500), Image.ANTIALIAS) #The (250, 250) is (height, width)
            img_2 = img_2.resize((380, 500), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(img) #llamar imagen
            self.photo_2 = ImageTk.PhotoImage(img_2)
            self.canvas1.create_image(800, 50, image=self.photo, anchor=NW)#abrir en ventana
            self.canvas1.create_image(50, 50, image=self.photo_2, anchor=NW)
            self.errores.set(" Imagen procesada :: ")
            return

    def Thresholding(self,image1):
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        res = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        h, w = img1.shape
        winsize = 14  #tamaño de la subventana para cada pixel
        const = 6 #constate
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
                if x1 > w:  #columnas
                    x1 = w
                block =img1[y0:y1, x0:x1]
                threshold =np.mean(block)-const # sacamos el promedio de la sub ventana restado por la constante
                if img1[i,j]<threshold:
                    res[i,j] = 0
                else:
                    res[i,j] = 255
        imgout = res
        kernel = np.ones((3,3), np.uint8)       # kernel como matrix 3x3 de
        erosion_image = cv2.erode(imgout, kernel, iterations=1)
        resultado = cv2.dilate(erosion_image, kernel, iterations=1)
        return resultado

    def Coef(self,a, b, n):
        result = []
        b = [b[0], b[1], 1]
        tam = 3
        for i in range(tam):
            A = [0]*tam*4
            A[i] = a[0]
            A[tam+i] = a[1]
            A[2*tam+i] = 1 if i != 2 else 0
            A[3*tam+n-1] = -b[i]
            result.append(A)
        return result

    def getPerspectiveTransform(self,pts1, pts2):
        A = []
        entrada= len(pts1)
        for i in range(entrada):
            A += self.Coef(pts1[i], pts2[i], i)#obtengo coeficionetes mediante los puntos de entrada
        B = [0, 0, -1] * entrada
        C = np.linalg.solve(A, B)
        res = np.ones(9)# Los primeros 8 elementos de C ahora contienen una matriz de transformación aplanada M (el noveno elemento siempre se establece en 1) 
        res[:8] = C.flatten()[:8] 
        return res.reshape(3,-1).T 

    def convertMatrix(self,img):
        h,w,c = img.shape
        matriz = np.zeros((w,h,c), dtype='int')
        for i in range(img.shape[0]):
            matriz[:,i] = img[i]
        
        return matriz

    def convertImage(self,matriz):
        w,h,c = matriz.shape
        img = np.zeros((h,w,c), dtype='int')
        for i in range(matriz.shape[0]):
            img[:,i] = matriz[i]
        return img

    def warpPerspective(self,img, M, dsize):
        matriz = self.convertMatrix(img)
        fil,col = dsize
        dst = np.zeros((fil,col,matriz.shape[2]))
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                res = np.dot(M, [i,j,1])
                i2,j2,_ = (res / res[2] + 0.5).astype(int)
                if i2 >= 0 and i2 < fil:
                    if j2 >= 0 and j2 < col:
                        dst[i2,j2] = matriz[i,j]
        return self.convertImage(dst)

    def AffineMejorado(self,M,img,hei,wei):
        X=[0,0]
        h,w,c=img.shape
        img_out=np.zeros((hei,wei,c),np.uint8)#Creamos nuestra matriz para la respuesta
        iden=np.array([[M[1][1],M[1][0]],[M[0][1],M[0][0]]])
        B=np.array([[M[1][2]],[M[0][2]]])
        for i in range(hei):#tomamos las dimensiones
            for j in range(wei):
                vector=np.array([[i],[j]])#aplicamos la operacion solve(A,Y,X)
                Y=vector-B
                X=cv.solve(iden,Y)
                x=int(X[1][0][0])
                y=int(X[1][1][0])
                if(x<img_out.shape[0] and x>=0):
                    if(y<img_out.shape[1] and y>=0):
                        img_out[i][j]=img[x][y]#colocaos en su pixel correspondiente
        return img_out

    def harris_corners(self,image):
        points=[]
        cornerness_measurements = []
        section_list = []
        corner_list = []
        height = image.shape[0]
        width = image.shape[1]
        height_section = height / 5
        width_section = width / 5
        partition_corners = copy.deepcopy(image)
        gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        direction_grad_im_x = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0, ksize=3)
        direction_grad_im_y = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1, ksize=3)
        i_x_x = direction_grad_im_x * direction_grad_im_x
        i_x_y = direction_grad_im_x * direction_grad_im_y
        i_y_y = direction_grad_im_y * direction_grad_im_y
        for j in range(1, height - 1):
            for i in range(1, width - 1):
                sum_i_x_x = 0
                sum_i_x_y = 0
                sum_i_y_y = 0
                for k in (-1, 0, 1):
                    for m in (-1, 0, 1):
                        sum_i_x_x += i_x_x[j+k][i+m]
                        sum_i_x_y += i_x_y[j+k][i+m]
                        sum_i_y_y += i_y_y[j+k][i+m]
                det = (sum_i_x_x * sum_i_y_y) - (sum_i_x_y ** 2)
                trace = sum_i_x_x + sum_i_y_y
                r = det - (.05 * trace)
                cornerness_measurements.append([i, j, r])
        cornerness_measurements = sorted(cornerness_measurements, key=itemgetter(2), reverse=True)
        for i in range(5):
            for j in range(5):
                section_list.clear()
                x1 = width_section * j
                x2 = width_section * (j + 1)
                y1 = height_section * i
                y2 = height_section * (i + 1)
                for k in cornerness_measurements:
                    if x1 <= k[0] < x2:
                        if y1 <= k[1] < y2:
                            section_list.append(k)
                points.append([section_list[m][0],section_list[m][1]])
        points_show=[]  
        points_sort_Y=[]
        points = sorted(points, key=itemgetter(0), reverse=False)
        max_X_proximo= points[len(points) - 2]
        points_show.append(points[len(points) - 1])
        min_X_proximo= points[1]
        points_show.append(points[0])
        points_sort_Y = sorted(points, key=itemgetter(1), reverse=False)
        max_Y = points_sort_Y[len(points_sort_Y) - 1]
        points_show.append( points_sort_Y[1])
        min_Y = points_sort_Y[0]
        points_show.append(points_sort_Y[0])
        for xy in range(4):
            cv2.circle(partition_corners, (points_show[xy][0], points_show[xy][1]), 6, (0, 0, 255), -1)
        cv2.imwrite("esquinas.png", partition_corners)
        #cv2.imshow("esquinas.png", partition_corners)
        return points_show

    def multiplicacion(self):
        if(self.auto):
            image1 = cv2.imread("esquinas.png")
        else:
            image1 = cv2.imread(self.img)
        c = self.c_d.get()
        img = image1.copy()    
        img = img.astype(int)
        h,w,canal = img.shape
        for i in range(canal):
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    tmp = (img[j][k][i]*c)
                    if (tmp >= 255 ):
                        img[j][k][i] = 255
                    else:
                        img[j][k][i] = tmp
        cv2.imwrite("add.png", img)
        img = Image.open("add.png")
        img = img.resize((380, 500), Image.ANTIALIAS) #The (250, 250) is (height, width)
        img_2 = img_2.resize((380, 500), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(img) #llamar imagen
        self.photo_2 = ImageTk.PhotoImage(img_2)
        self.canvas1.create_image(800, 50, image=self.photo, anchor=NW)#abrir en ventana
        self.canvas1.create_image(50, 50, image=self.photo_2, anchor=NW)
        self.errores.set(" Imagen procesada :: ")
        return

    def procesar_colores(self,image1):
        img = image1.copy()    
        img = img.astype(int)
        h,w,canal = img.shape
        for i in range(canal):
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    tmp =(img[j][k][i]*1.7)
                    if (tmp >= 255 ):
                        img[j][k][i] = 255
                    else:
                        img[j][k][i] = tmp
        res = img
        return res
    def gray(self,image):
        gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_im


def main():
    mi_app = Aplicacion()
    return 0

if __name__ == '__main__':
    main()