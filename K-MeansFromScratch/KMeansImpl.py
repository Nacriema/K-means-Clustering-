'''
Implementation of K-Means Cluster

Thuật toán như sau:

1. Đầu tiên ta khởi tạo k điểm ngẫu nhiên, được gọi là các means
2. Sau đó ta phân loại các điểm trong tập hợp điểm cần phân loại về điểm mean nào gần với nó nhất.
3. Ta cập nhật lại vị tr ícho các tâm cụm means của chúng ta bằng hàm tính trung bình thôi.
4. Lặp lại bước 2 -> 3 cho đến khi một điều kiện ràng buộc được thỏa mãn

Appendix: Điều kiện ràng buộc tại bước 4 có thể là: đạt tới số lần lặp tối đa mà mình quy định trước, có thể là khi nào
mà các cụn không còn sự thay đổi nữa <=> không còn sự di chuyển các điểm từ nhóm này sang nhóm khác.

Cái K-Means ở đây mình dùng để áp dụng cho bải toán nén ảnh, mình chuyển tấm ảnh đó thành ảnh 8 màu thôi

À mà cái Kmeans trong Sklearn thì nó support cho dữ liệu dạng (samples, features) thôi nha.
'''

import numpy as np
from PIL import Image
import os


def read_image(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image = Image.open(dir_path + '/' + name)
    image = np.array(image)
    return image


def initialize_means(img, cluster):
    '''
    :param img:
    :param cluster:
    :return: Trả về 2 thứ là các điểm ảnh và giá trị của k cluster means
    '''
    # reshape ảnh về (samples, features)
    points = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    samples, features = points.shape

    # means la mang cua mean cac centroid
    means = np.zeros((cluster, features))

    # Random khoi tao gia tri cho cac means
    for _ in range(cluster):
        rand = np.random.randint(samples)
        means[_] = points[rand]
    return points, means


# Ham su dung khoang cach Euclide de uoc luong
def distance(p1, p2):
    '''Tinh khoảng cách euclide giữa 2 điểm p1 và p2 (thực ra mỗi điểm chỉ có 3 features thôi) '''
    dist = np.sqrt(np.sum(np.square(p1 - p2)))
    return dist


def k_means(points, means, clusters, n_iter=100):
    samples, features = points.shape

    # Gia tri index ma moi diem no thuoc vao nhom nao
    index = np.zeros(samples)

    # K-Means algorithm
    while n_iter > 0:
        # Xet khoang cach moi diem trong tap points toi tung means, thang nao nho hon thi lay chi so cua mean do
        for j in range(len(points)):
            minV = 1000000

            for k in range(clusters):
                x1 = points[j]
                x2 = means[k]

                if distance(x1, x2) < minV:
                    minV = distance(x1, x2)
                    index[j] = k

        # Cap nhat lai trong so cho moi thang means
        for k in range(clusters):
            sum = np.zeros(features)
            count = 0
            for j in range(len(points)):
                if index[j] == k:
                    count += 1
                    sum = np.add(sum, points[j])
            means[k] = sum / count
        n_iter -= 1
    return means, index


def compress_image(means, index, img):
    '''Dua tren nhung thong so sau khi phan loại, hinh thanh lai buc hinh'''
    imgShape = img.shape
    centroid = np.array(means)

    # Tái tạo lại 2D hình ảnh
    img_arr = [means[i] for i in(index.astype(int))]
    img_arr = np.array(img_arr)
    img_arr = img_arr.astype("uint8")

    img_rs = img_arr.reshape(imgShape)
    print('Chiều của ảnh mới ra:', img_rs.shape)
    img_end = Image.fromarray(img_rs, 'RGB')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_end.save(dir_path + '/outCat.jpg')
    img_end.show()


if __name__ == '__main__':
    print(__doc__)

    img = read_image('cat.jpg')
    clusters = 8
    points, means = initialize_means(img, clusters)
    print('Số điểm ảnh: ', len(points))
    print('Một vài điểm ảnh: ', points[1: 10, :])
    print('Means ban đầu: ', means)
    means, index = k_means(points, means, clusters, n_iter=5)
    print('Means lúc sau: ', means)
    print('Class của các điểm: ', index)
    (u, c) = np.unique(index, return_counts=True)
    freq = np.array((u, c)).T
    print('Sô loại điểm ảnh, và tần số: ')
    print(freq)

    # Nén và hiển thị hình ảnh
    compress_image(means, index, img)