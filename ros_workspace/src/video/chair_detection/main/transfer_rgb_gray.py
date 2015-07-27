import numpy as np
import cv2

def load_data(image):
        # data_x=np.load("data_x.npy")

        data_x_new=[]
        for img in data_x:
            img= cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
            data_x_new.append(img)
        data_x_new = np.asarray(data_x_new)
        np.save("data_x_new.npy", data_x_new)


load_data()