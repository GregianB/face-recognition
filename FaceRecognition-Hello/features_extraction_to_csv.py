import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Path of cropped faces
path_images_from_camera = "dataset/"

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")

# Function to apply histogram equalization
def histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

# Function to apply data augmentation (e.g., rotation)
def augment_image(image):
    # Rotate image randomly by a few degrees
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented_image = cv2.warpAffine(image, matrix, (w, h))
    return augmented_image

# Return 128D features for single image
def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)

    # Apply histogram equalization and augmentation
    img_rd = histogram_equalization(img_rd)
    img_rd = augment_image(img_rd)

    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", " Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face detected")
    return face_descriptor

# Return the mean value of 128D face descriptor for person X
def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # Get 128D features for single image of personX
            logging.info("%-40s %-20s", " / Reading image:", path_face_personX + "/" + photos_list[i])
            features_128d = return_128d_features(path_face_personX + "/" + photos_list[i])
            # Jump if no face detected from image
            if features_128d == 0:
                continue  # Skip this iteration if no face detected
            else:
                features_list_personX.append(features_128d)
    else:
        logging.warning(" Warning: No images in %s/", path_face_personX)

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX

# Function to check existing names in CSV
def read_existing_names(csv_file):
    existing_names = set()
    if os.path.exists(csv_file):
        with open(csv_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    existing_names.add(row[0])  # Assuming first column is the person's name
    return existing_names

def main():
    logging.basicConfig(level=logging.INFO)
    # Get the order of latest person
    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    # Ensure the directory exists
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Check existing names in CSV
    csv_file = "data/features_all.csv"
    existing_names = read_existing_names(csv_file)

    with open(csv_file, "a", newline="") as csvfile:  # Use "a" mode to append
        writer = csv.writer(csvfile)
        for person in person_list:
            # Get the person's name
            if len(person.split('_', 2)) == 2:
                # "person_x"
                person_name = person
            else:
                # "person_x_tom"
                person_name = person.split('_', 2)[-1]

            # Skip if person already exists in CSV
            if person_name in existing_names:
                logging.info("Person %s already exists, skipping...", person_name)
                continue

            # Get the mean/average features of face/personX
            logging.info("%sperson_%s", path_images_from_camera, person)
            features_mean_personX = return_features_mean_personX(path_images_from_camera + person)

            # Insert person's name to features and write to CSV
            features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
            writer.writerow(features_mean_personX)
            logging.info('\n')
        logging.info("Saved all the features of faces registered into: data/features_all.csv")

if __name__ == '__main__':
    main()
