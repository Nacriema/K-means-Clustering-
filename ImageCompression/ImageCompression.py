# Chu minh lam ro so sanh thu voi ket qua minh dat duoc, tach lop ra, xong roi minh Kmeans 8 cum moi lop, sau do ep no
# vao lai

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
image1 = Image.open(dir_path + '/cat.jpg')
# image1 = Image.open('cat.jpg')
image1.show()
image2 = np.array(image1)
shapeRoot = image2.shape

# Lay 3 kenh mau cua no ra, moi kenh mau kmeans ra 2 cum
image2_red = image2[:, :, 0]
image2_green = image2[:, :, 1]
image2_blue = image2[:, :, 2]

print(image2_red.shape)

test = Image.fromarray(image2_red)
test.show()

# reshape image to 2d array
shapeImage2 = image2_red.shape
print('Shape Image Channel Red: ', shapeImage2)
image3_red = image2_red.reshape((-1, 1))
image3_green = image2_green.reshape((-1, 1))
image3_blue = image2_blue.reshape((-1, 1))

print(image3_red.shape)
print(image3_red)
print(type(image3_red))

kmeans_red = KMeans(n_clusters=2, max_iter=300, random_state=0).fit(image3_red)
kmeans_green = KMeans(n_clusters=2, max_iter=300, random_state=0).fit(image3_green)
kmeans_blue = KMeans(n_clusters=2, max_iter=300, random_state=0).fit(image3_blue)


print('Nhan: ', len(kmeans_red.labels_))
print('Centroids: ', kmeans_red.cluster_centers_)

# Xong roi, gio thi minh lam cai veo chi day
rs_list_red = [kmeans_red.cluster_centers_[i] for i in (kmeans_red.labels_)]
rs_list_red = np.array(rs_list_red)

rs_list_green = [kmeans_green.cluster_centers_[i] for i in (kmeans_green.labels_)]
rs_list_green = np.array(rs_list_green)

rs_list_blue = [kmeans_blue.cluster_centers_[i] for i in (kmeans_blue.labels_)]
rs_list_blue = np.array(rs_list_blue)

print(type(rs_list_red))
print(rs_list_red.shape)

image_result_red = rs_list_red.reshape(shapeImage2)
print(image_result_red.shape)

image_result_green = rs_list_green.reshape(shapeImage2)
print(image_result_green.shape)

image_result_blue = rs_list_blue.reshape(shapeImage2)
print(image_result_blue.shape)



img_out_red = Image.fromarray(image_result_red)
#img_out.save('outimg.jpg')
img_out_red.show()

img_out_green = Image.fromarray(image_result_green)
#img_out.save('outimg.jpg')
img_out_green.show()

img_out_blue = Image.fromarray(image_result_blue)
#img_out.save('outimg.jpg')
img_out_blue.show()

img_end = np.dstack((image_result_red, image_result_green, image_result_blue))
img_end = img_end.astype("uint8")

rs_list = img_end.reshape((shapeRoot[0] * shapeRoot[1]), shapeRoot[2])
# print(img_end)
# (unique, count) = np.unique(img_end, return_counts=True)
# print(np.asarray((unique, count)).T)
# print('Chieu Image cuoi: ', img_end.shape)
img_end_end = Image.fromarray(img_end, 'RGB')

unique_rows = np.unique(rs_list, axis=0)
print('Danh sách màu trong bức ảnh: ')
print(unique_rows)
print('So mau trong tam anh: ', len(unique_rows))

img_end_end.show()