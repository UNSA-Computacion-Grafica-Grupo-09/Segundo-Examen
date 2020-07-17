import cv2
import copy
from operator import itemgetter


def harris_corners(image):
    points=[]
    cornerness_measurements = []
    section_list = []
    corner_list = []
    height = image.shape[0]
    width = image.shape[1]
    height_section = height / 5
    width_section = width / 5
    partition_corners = copy.deepcopy(image)

    #Convertimos a escala de grises
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Gradiente e X y Y
    direction_grad_im_x = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0, ksize=3)
    direction_grad_im_y = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1, ksize=3)

    # Determinamos i_x_x, i_x_y, y i_y_y para los calculos
    i_x_x = direction_grad_im_x * direction_grad_im_x
    i_x_y = direction_grad_im_x * direction_grad_im_y
    i_y_y = direction_grad_im_y * direction_grad_im_y

     
    for j in range(1, height - 1):
        for i in range(1, width - 1):

            #Encontrar las sumas de IxIx, IxIy e IyIy para cada píxel y 8 circundantes
            sum_i_x_x = 0
            sum_i_x_y = 0
            sum_i_y_y = 0
            for k in (-1, 0, 1):
                for m in (-1, 0, 1):
                    sum_i_x_x += i_x_x[j+k][i+m]
                    sum_i_x_y += i_x_y[j+k][i+m]
                    sum_i_y_y += i_y_y[j+k][i+m]

            #Use sumas para calcular determinantes, trazas y r
            det = (sum_i_x_x * sum_i_y_y) - (sum_i_x_y ** 2)
            trace = sum_i_x_x + sum_i_y_y
            r = det - (.05 * trace)

            #Crear una lista de medidas de esquina desde todas las ubicaciones de píxeles y valores r
            cornerness_measurements.append([i, j, r])

    #Ordenar cornerMeasurements
    cornerness_measurements = sorted(cornerness_measurements, key=itemgetter(2), reverse=True)
    #print(cornerness_measurements) [178, 394, -2656.8],

    
    
    for i in range(5):
        for j in range(5):
            section_list.clear()

            #Calcular las secciones de las coordenadas
            x1 = width_section * j
            x2 = width_section * (j + 1)
            y1 = height_section * i
            y2 = height_section * (i + 1)
            
             
            #Determinar si las medidas de esquina están en la sección
            for k in cornerness_measurements:
                if x1 <= k[0] < x2:
                    if y1 <= k[1] < y2:
                        section_list.append(k)

            points.append([section_list[m][0],section_list[m][1]])

            # Dibuja círculos verdes para denotar n píxeles con los valores de esquina más altos en el vecindario
            # cornernessMeasurements ya se ordenó, por lo que section_list contiene las medidas de esquina más altas en
            # frente de la lista

    points_show=[]  
    points_sort_Y=[]
    points = sorted(points, key=itemgetter(0), reverse=False)
    print("ordenados en X")
    print(points)
    # max y min en X
    max_X_proximo= points[len(points) - 2]
    points_show.append(points[len(points) - 1])
    min_X_proximo= points[1]
    points_show.append(points[0])
   
    points_sort_Y = sorted(points, key=itemgetter(1), reverse=False)
    print("ordenados en Y")
    print(points_sort_Y)
    max_Y = points_sort_Y[len(points_sort_Y) - 1]
    points_show.append( points_sort_Y[1])
    min_Y = points_sort_Y[0]
    points_show.append(points_sort_Y[0])
    print(points_show)

    for xy in range(4):
        cv2.circle(partition_corners, (points_show[xy][0], points_show[xy][1]), 6, (0, 0, 255), -1)
 

    cv2.imwrite("esquinas.png", partition_corners)
    cv2.imshow("esquinas.png", partition_corners)
    return points_show

      
def main():
    imageIn = "hoja3.png"
    image = cv2.imread(imageIn)
    points = harris_corners(image)
    cv2.waitKey(0)

main()
