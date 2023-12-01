import torch
import os
import cv2
from time import time
import numpy as np


def anonymize_license_plate(image, factor=3.0):
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1
    return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_license_plate_pixelate(image, blocks=3):
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)
    return image


def blurr_images(model, input_path="../test", output_path="results", batch_size=10):
    images = [f"{input_path}/{f}" for f in os.listdir(input_path)]

    results_dir_num = 0
    while f"exp{results_dir_num}" in os.listdir(output_path):
        results_dir_num += 1
    os.mkdir(f"{output_path}/exp{results_dir_num}")

    start = time()

    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]

        results = model(batch_images)

        for i, res_df in enumerate(results.pandas().xyxy):
            img_idx = start_idx + i
            img_path = images[img_idx]
            img = cv2.imread(img_path)

            for j in range(len(res_df)):
                xmin = round(res_df['xmin'][j])
                ymin = round(res_df['ymin'][j])
                xmax = round(res_df['xmax'][j])
                ymax = round(res_df['ymax'][j])
                license_plate = img[ymin:ymax, xmin:xmax]
                img[ymin:ymax, xmin:xmax] = anonymize_license_plate_pixelate(license_plate, 10)

            output_folder = f"{output_path}/exp{results_dir_num}"
            output_filename = f"output{img_idx}.jpg"
            cv2.imwrite(os.path.join(output_folder, output_filename), img)

    end = time()
    timer = end - start
    print('Czas:', timer, 'sekund')


def blurr_video(model, input_path, output_path="results"):
    video = cv2.VideoCapture(input_path)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    results_dir_num = 0
    while f"exp{results_dir_num}" in os.listdir("results"):
        results_dir_num += 1
    os.mkdir(f"{output_path}/exp{results_dir_num}")

    output = cv2.VideoWriter(f"{output_path}/exp{results_dir_num}/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    while True:
        ret, frame = video.read()
        if ret:
            result = model(frame)
            res_df = result.pandas().xyxy[0]
            for j in range(len(res_df)):
                if res_df['confidence'][j] < 0.2:
                    continue
                xmin = round(res_df['xmin'][j])
                ymin = round(res_df['ymin'][j])
                xmax = round(res_df['xmax'][j])
                ymax = round(res_df['ymax'][j])
                license_plate = frame[ymin:ymax, xmin:xmax]
                frame[ymin:ymax, xmin:xmax] = anonymize_license_plate_pixelate(license_plate, 10)   # change
            output.write(frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    cv2.destroyAllWindows()
    video.release()
    output.release()


if __name__ == "__main__":
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\mateu\\Desktop\\bluring_plates\\yolov5\\runs\\train\\exp61\\weights\\best.pt')
    blurr_images(model=yolo_model, input_path="C:\\Users\\mateu\\Desktop\\mag\\Rejestracje_testy", output_path="C:\\Users\\mateu\\Desktop\\mag\\wyniki", batch_size=10)
    # blurr_video(model=yolo_model, input_path="test_videos/test3.mp4", output_path="results")