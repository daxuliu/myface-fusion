# faceswap


- 运行 video.py 的主函数
```
python video.py
```


```
core.face_merge(src_img='images/model.jpg',
                    dst_img='images/20171030175254.jpg',
                    out_img='images/output.jpg',
                    face_area=[50, 30, 500, 485],
                    alpha=0.75,
                    blur_size=(15, 10),
                    mat_multiple=0.95)
```
参数含义：
- src_img —— 模特图片
- dst_img —— 待融合的图片
- out_img —— 结果图片输出路径
- face_area —— 指定模板图中进行人脸融合的人脸框位置。四个正整数数组，依次代表人脸框左上角纵坐标（top），左上角横坐标（left），人脸框宽度（width），人脸框高度（height），通过设定改参数可以减少结果的大范围变形，把变形风险控制在人脸框区域
- alpha —— 融合比例，范围 [0,1]。数字越大融合结果包含越多融合图 (dst_img) 特征。
- blur_size—— 模糊核大小，用于模糊人脸融合边缘，减少融合后的违和感
- mat_multiple —— 缩放获取到的人脸心型区域

### 一、 检测及关键的定位

人脸的检测以及关键点定位有多种实现方案

- 使用开源 Dlib 库检测及定位（定位68个关键点）
- 使用腾讯平台的人脸识别及定位API （定位90个关键点）
- 使用Face++平台的人脸识别定位API（定位106个关键点）

本文采用的是Face++的 api，因为商用情况下 Face++ 定位的定数最多

```
// 获取两张图片的人脸关键点（矩阵格式与数组格式）
src_matrix, src_points, err = core.face_points(src_img)
dst_matrix, dst_points, err = core.face_points(dst_img)
```

### 二、对齐人脸角度

在待融合图人像不是侧脸的情况下，我们可以同过调整平面位置及角度让其与模特图的人脸重合

```
    // opencv 读取图片
    src_img = cv2.imread(src_img, cv2.IMREAD_COLOR)
    dst_img = cv2.imread(dst_img, cv2.IMREAD_COLOR)

    dst_img = transformation_points(src_img=src_img, src_points=src_matrix[core.FACE_POINTS],
                                    dst_img=dst_img, dst_points=dst_matrix[core.FACE_POINTS])
```
* 注：src_points 已经 dst_points 传入参数为第一步获取的人脸关键点矩阵

对齐采用“常规 Procrustes 分析法”

具体算法来源：[matthewearl](http://matthewearl.github.io/2015/07/28/switching-eds-with-python/) 个人博客步骤2

对齐结果：

![结果展示](/images/2282038-87b0d91ba49136da.gif)

### 三、再次取点后融合脸部

对步骤二转换后的带融合图片再次取关键的，然后与模特图的关键点一起做三角融合成新的图片
```
dst_img = morph_img(src_img, src_points, dst_img, dst_points, alpha)
```
融合结果：

![结果展示](/images/2282038-6f62bb9178d8ea54.jpg)

具体的三角融合算法解说参考[这篇文章](https://www.learnopencv.com/face-morph-using-opencv-cpp-python/)

### 四、处理加工模特图片

再次对上一步的结果图进行取点，然后运用三角仿射将模特图片脸部轮廓、关键点变形成上一步得到的脸部关键点
```
src_img = tran_src(src_img, src_points, dst_points, face_area)
```
处理结果：

![结果展示](/images/2282038-13d70b5c2508afda.jpg)


### 五、将融合后的脸部贴到模特图上

最后一步是将融合后的新图片脸部区域用泊松融合算法贴到模特图上。泊松融合可直接使用opencv提供的函数
```
dst_img = merge_img(src_img, dst_img, dst_matrix, dst_points, k_size, mat_multiple)
```

```
def merge_img(src_img, dst_img, dst_matrix, dst_points, k_size=None, mat_multiple=None):
    face_mask = np.zeros(src_img.shape, dtype=src_img.dtype)

    for group in core.OVERLAY_POINTS:
        cv2.fillConvexPoly(face_mask, cv2.convexHull(dst_matrix[group]), (255, 255, 255))

    r = cv2.boundingRect(np.float32([dst_points[:core.FACE_END]]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    if mat_multiple:
        mat = cv2.getRotationMatrix2D(center, 0, mat_multiple)
        face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

    if k_size:
        face_mask = cv2.blur(face_mask, k_size, center)

    return cv2.seamlessClone(np.uint8(dst_img), src_img, face_mask, center, cv2.NORMAL_CLONE)
```
函数示意图：

![步骤展示](/images/2282038-362be008f850ba22.jpg)

