import cv2

import cv2
import os
from PIL import Image
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


def PicToVideo(imgPath, videoPath,model=""):
    images = os.listdir(imgPath)
    fps = 25  # 帧率
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    im = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, im.size)
    for im_name in range(len(images)):
        faca_area = getFaceAres(images[im_name])
        core.face_merge(src_img=images[im_name],
                        dst_img=model,
                        out_img="out"+images[im_name],
                        face_area=faca_area,
                        alpha=0.75,
                        k_size=(15, 10),
                        mat_multiple=0.95)
        frame = cv2.imread(imgPath + "out"+images[im_name])
        videoWriter.write(frame)
        print(im_name)
    videoWriter.release()


imgPath = "img1/"
videoPath = "video2.mp4"
PicToVideo(imgPath, videoPath)


def videoToImg(name=""):
    vc=cv2.VideoCapture(name)
    c = 0
    rval = vc.isOpened()

    while rval:
        c = c + 1
        rval, frame = vc.read()
        print(c)
        if rval:
            cv2.imwrite("img1/"+name.split(".")[0]+ str(c) + '.png', frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()
# videoToImg("IMG_3566.MOV")
if __name__ == '__main__':
    videoToImg("IMG_3566.MOV")
    imgPath = "img1/"
    videoPath = "outVideo.mp4"
    PicToVideo(imgPath, videoPath,"jobs.jpg")

