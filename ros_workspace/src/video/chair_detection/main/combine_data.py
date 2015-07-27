import numpy as np



def load_data():
        data_x=np.load("data_x_new.npy")
        data_y=np.load("data_y.npy")
        data_x1=np.load("data_x_new1.npy")
        data_y1=np.load("data_y1.npy")
        print "datax", data_x.shape
        print "datax1", data_x1.shape


        data_x2 = np.concatenate((data_x, data_x1), axis = 0)
        data_y2 = np.concatenate((data_y, data_y1), axis = 0)

        print "datax", data_x2.shape

        np.save("data_x3.npy", data_x2)
        np.save("data_y3.npy", data_y2)

if __name__ == "__main__":
        load_data()
