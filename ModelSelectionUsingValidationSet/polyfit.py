import numpy as np
import matplotlib.pyplot as plt


#Load train and validation data
train = np.loadtxt('data_training.txt')
valid = np.loadtxt('data_validation.txt')

x_train = train[:,0]
y_train = train[:,1]

x_valid = valid[:,0]
y_valid = valid[:,1]


#It is convenient to use poly1d objects for dealing with polynomials:
z = np.polyfit(x_train,y_train,5)       # for the illusturation purposes
p = np.poly1d(z)

x = train[:,0]
y = train[:,1]

t = np.linspace(-8, 13, 200)
plt.plot(x, y, '.', t, p(t), '-')
plt.show()

def calculate_error(x,y,p):     #x, y coordinates and p function from poly1d
    error = 0
    sum = 0
    for i in range(len(y)):
        sum = sum + (y[i] - np.polyval(p,x[i]))**2
    error = sum / len(y)
    return error


test_error = []
valid_error = []

for m in range(1,21):
    #for train data fit
    z = np.polyfit(x_train,y_train,m)
    p = np.poly1d(z)
    error_train = calculate_error(x_train,y_train,p)
    error_valid = calculate_error(x_valid,y_valid,p)
    test_error.append(error_train)
    valid_error.append(error_valid)


pts = np.arange(1,21,1)
plt.plot(pts,test_error,'-b',pts,valid_error,'-g')
plt.scatter(pts,valid_error, color='r', s=10)
plt.scatter(pts,test_error, color='r', s=10)
plt.xlabel('Polynomial Degree')
plt.ylabel('Fit Error(RMS)')
plt.show()

#Automatically pick polynomial with lowest validation error and write the polynomial,
min_index = valid_error.index(min(valid_error))+1 #returns the index of  minimum value of the errors    +1 for the actual value
z = np.polyfit(x_train, y_train, min_index)
p = np.poly1d(z)
print("Printing min value of validation error",valid_error.index((min(valid_error))))
print(p)

