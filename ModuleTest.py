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
    src="images/IMG_0627.JPG"
    des="images/IMG_3572.jpg"
    out="out1"
    faca_area=getFaceAres(src)
    area2=getFaceAres(des)
    print(faca_area,area2)
    print(area2[-2]/faca_area[-2],area2[-1]/faca_area[-1])
    core.face_merge(src_img=src,
                    dst_img=des,
                    out_img='images/'+out+'.jpg',
                    face_area=getFaceAres(src),
                    alpha=getFaceAres(des)[-2]/getFaceAres(src)[-2],
                    k_size=(5,5 ),
                    mat_multiple=getFaceAres(des)[-2]/getFaceAres(src)[-2])
