import imgaug.augmenters as iaa 
import cv2 
import matplotlib.pyplot as plt 

image = cv2.imread('./img_1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(0)
plt.imshow(image)

snow_value = 0
aug = iaa.Rain(drop_size=(0.02, 0.02), speed=(0.1,0.1), nb_iterations=(2,2), density=(0.0,0.0))
image_aug = aug(image=image)
plt.figure(1)
plt.imshow(image_aug)

snow_value = 0
aug = iaa.Rain(drop_size=(0.02, 0.02), speed=(0.1,0.1), nb_iterations=(2,2), density=(0.1,0.1))
image_aug = aug(image=image)
plt.figure(2)
plt.imshow(image_aug)

snow_value = 0
aug = iaa.Rain(drop_size=(0.02, 0.02), speed=(0.1,0.1), nb_iterations=(2,2), density=(0.3,0.3))
image_aug = aug(image=image)
plt.figure(3)
plt.imshow(image_aug)

snow_value = 0
aug = iaa.Rain(drop_size=(0.02, 0.02), speed=(0.1,0.1), nb_iterations=(2,2), density=(0.5,0.5))
image_aug = aug(image=image)
plt.figure(4)
plt.imshow(image_aug)


plt.show()
