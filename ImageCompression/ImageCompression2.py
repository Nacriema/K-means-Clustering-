from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
image = Image.open(dir_path + '/cat.jpg')
image.show()
image = np.array(image)
shapeRoot = image.shape


# Reshape lai cai mang
image_feed = image.reshape((shapeRoot[0] * shapeRoot[1]), shapeRoot[2])
kmeans = KMeans(n_clusters=8, max_iter=300, random_state=0).fit(image_feed)

# Xay dung lai cai np array theo ket qua
rs_list = [kmeans.cluster_centers_[i] for i in (kmeans.labels_)]
rs_list = np.array(rs_list)
rs_list = rs_list.astype("uint8")

unique_rows = np.unique(rs_list, axis=0)
print('Danh sách màu trong bức ảnh: ')
print(unique_rows)
print('So mau trong tam anh: ', len(unique_rows))

rs_list = rs_list.reshape(shapeRoot)
print(rs_list.shape)

img_end = Image.fromarray(rs_list, 'RGB')
img_end.save('outimg2.jpg')
img_end.show()



