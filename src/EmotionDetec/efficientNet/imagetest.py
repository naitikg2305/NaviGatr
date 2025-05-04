import numpy as np 
import cv2


image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/20250312_225342.jpg'  # Update this with an actual image path
image_original = cv2.imread(image_path)
image_original_array = np.array(image_original)
image = cv2.resize(image_original, (224, 224))
image_array = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

print(image_original.shape) #should be 2544 x 3392
print(image_original_array.shape) #should be 2544 x 3392
print(image_array.shape)# this should be 224 X 224


print(type(image_original)) #should be 2544 x 3392
print(type(image_original_array)) #should be 2544 x 3392
print(type(image_array))# this should be 224 X 224

#