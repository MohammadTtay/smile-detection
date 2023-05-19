import cv2
import os
import numpy as np

images_dir = os.path.dirname(os.path.abspath(__file__))+"/files/"
cropped_dir = os.path.dirname(os.path.abspath(__file__))+"/cropped_files/"

if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):

            img_path = os.path.join(images_dir, filename)
            img = cv2.imread(img_path)

            # Detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)

            # Crop and save faces
            for (x, y, w, h) in faces:
                cropped_img = img[y:y+h, x:x+w]
                image = cv2.resize(cropped_img, (200, 200))
                cropped_filename = 'cropped_' + filename
                cropped_path = os.path.join(cropped_dir, cropped_filename)
                cv2.imwrite(cropped_path, image)

            # Remove The second and third columns in labels.txt
            labels = np.loadtxt('labels.txt', delimiter=' ').astype(int)[:, 0].reshape(-1, 1)
            np.savetxt(f'{cropped_dir}new_labels.txt', labels, delimiter=' ', fmt='%i')
