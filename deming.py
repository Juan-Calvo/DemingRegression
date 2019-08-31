import numpy as np
import matplotlib.pyplot as plt
 
class SimpleLinearRegression(object):
    def __init__(self):
        self.w = np.zeros(2)
 
    def train(self, x_data, y_data):
        delta=0.9
        n = len(x_data)

        sum_x = np.sum(x_data)
        sum_y = np.sum(y_data)

        av_x = sum_x/n
        av_y = sum_y/n

        sxx = np.sum(np.power((x_data-av_x),2))*(1/(n-1))
        syy = np.sum(np.power((y_data-av_y),2))*(1/(n-1))
        sxy = np.sum((x_data-av_x).dot(y_data-av_y))*(1/(n-1))
    
        self.w[1]=np.divide(syy - delta*sxx + np.sqrt(np.power((syy - delta*sxx),2) + 4*delta*np.power(sxy,2)),(2*sxy))
        self.w[0]= (av_y - self.w[1]*av_x)

    def predict_x(self, y):
        return np.true_divide((y-self.w[0]),self.w[1])

    def predict_y(self, x):
        return np.array([1, x]).dot(self.w)

    def get_intercept(self):
        return self.w[0]

    def get_slope(self):
        return self.w[1]
 

file = open ('your_file.txt','r')
l = []
l = [ line.split() for line in file]
l= np.array(l)
x=l[:,2]
y=l[:,3]


x_int=x.astype(np.float)
y_int=y.astype(np.float)


slr = SimpleLinearRegression()
slr.train(x_int, y_int)

plt.scatter(x_int, y_int)
plt.xlabel('title_x')
plt.ylabel('title_y')
plt.plot(slr.predict_x(y_int),y_int , color='red')
plt.show()
