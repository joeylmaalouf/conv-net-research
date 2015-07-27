import numpy as np 


def one_hot(x,n):
        if type(x) == list:
                x = np.array(x)
        x = x.flatten()
        o_h = np.zeros((len(x),n))
        for i in range(len(x)):
                o_h[i, int(bool(x[i]))] = 1
#       o_h[np.arange(len(x)),x] = 1
        return o_h

def form_square(data):
        l=[]
        for i in range(len(data)):
                (w,h)=data[i].shape
                if w > h:
                        offset = (w-h)/2
                        tmp = data[i][offset:w-offset, :]
                elif w < h:
                        offset = (h-w)/2
                        tmp = data[i][:, offset:h-offset]
                l.append(tmp)
        l=np.asarray(l)
        return l


def load_data():
        data_x=np.load("data_x.npy")
        data_y=np.load("data_y.npy")    


        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
        

        return trX,trY,teX,teY



if __name__ == "__main__":
        load_data()