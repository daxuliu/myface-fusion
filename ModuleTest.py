# -*- coding: utf-8 -*-
# @Time    : 2021/8/23
# @Author  : xuliu
import face_recognition
import core

def getFaceAres(img=""):
    image1 = face_recognition.load_image_file(img)
    arr=list(face_recognition.face_locations(image1)[0])
    print("arr",arr)
    area=[]
    area.append(arr[-1])
    area.append(arr[0])
    area.append(arr[1]-arr[-1])

    area.append(arr[2]-arr[0])
    print("area",area)
    return area

# [top, right, bottom, left] 分别代表框住人脸的矩形中左上角和右下角的坐标（x1,y1,x2,y2）
# [50, 30, 500, 485]
# 依次代表人脸框左上角纵坐标（top）y1，左上角横坐标（left）x1，人脸框宽度（width），人脸框高度（height），通过设定改参数可以减少结果的大范围变形，把变形风险控制在人脸框区域
if __name__ == '__main__':
    faca_area=getFaceAres("images/IMG_3572.JPG")
    core.face_merge(src_img='images/IMG_3572.JPG',
                    dst_img='images/IMG_3570.jpg',
                    out_img='images/outcook.jpg',
                    face_area=faca_area,
                    alpha=0.85,
                    k_size=(10, 5),
                    mat_multiple=0.75)
