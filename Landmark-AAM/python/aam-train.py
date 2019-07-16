import os
import time
import cv2
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of aam facial landmark algorithm.')
    parser.add_argument('--face_cascade', type=str, help="Path to the cascade model file for the face detector",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','haarcascade_frontalface_alt2.xml'))
    parser.add_argument('--aam_model', type=str, help="Path to save the aam trained model file",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','AAM_train.yaml'))
    parser.add_argument('--training_images', type=str, help="Path of a text file contains the list of paths to all training images",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train','images_train.txt'))
    parser.add_argument('--training_annotations', type=str, help="Path of a text file contains the list of paths to all training annotation files",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train','points_train.txt'))
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    start = time.time()
    facemark = cv2.face.createFacemarkAAM()
    if args.verbose:
        print("Creating the facemark took {} seconds".format(time.time()-start))

    start = time.time()
    status, images_train, landmarks_train = cv2.face.loadDatasetList(args.training_images, args.training_annotations)
    assert(status == True)
    if args.verbose:
        print("Loading the dataset took {} seconds".format(time.time()-start))

    scale = np.array([2.0, 4.0])
    facemark.setParams(args.face_cascade, args.aam_model, None, scale)

    for i in range(len(images_train)):
        start = time.time()
        img = cv2.imread(images_train[i])
        if args.verbose:
            print("Loading the image took {} seconds".format(time.time()-start))

        start = time.time()
        status, facial_points = cv2.face.loadFacePoints(landmarks_train[i])
        assert(status == True)
        if args.verbose:
            print("Loading the facepoints took {} seconds".format(time.time()-start))

        start = time.time()
        facemark.addTrainingSample(img,facial_points)
        assert(status == True)
        if args.verbose:
            print("Adding the training sample took {} seconds".format(time.time()-start))

    start = time.time()
    facemark.training()
    if args.verbose:
        print("Training took {} seconds".format(time.time()-start))