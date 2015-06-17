>>> img = cv2.imread('fly.png',0)
>>> surf = cv2.SURF(400)
>>> kp, des = surf.detectAndCompute(img,None)
>>> surf.hessianThreshold = 50000
>>> kp, des = surf.detectAndCompute(img,None)
>>> img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
>>> plt.imshow(img2),plt.show()
>>> print surf.upright
False
>>> surf.upright = True
>>> kp = surf.detect(img,None)
>>> img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
>>> plt.imshow(img2),plt.show()
 	