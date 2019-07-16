import os
import time
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstration of lbf facial landmark algorithm.')
    parser.add_argument('--face_cascade', type=str, help="Path to the cascade model file for the face detector",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','haarcascade_frontalface_alt2.xml'))
    parser.add_argument('--lbf_model', type=str, help="Path to the lbf trained model file",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','LBF.yaml'))
    parser.add_argument('--test_images', type=str, help="Path of the file containing the test images", default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'test','images_test.txt'))
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    face_cascade = cv2.CascadeClassifier(args.face_cascade)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(args.lbf_model)

    start = time.time()
    status, images_test, _ = cv2.face.loadDatasetList(args.test_images, args.test_images)
    assert(status == True)
    if args.verbose:
        print("Loading the dataset took {} seconds".format(time.time()-start))

    for image in images_test:
        frame = cv2.imread(image)
        if frame is not None :
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            if len(faces) > 0:
                start = time.time()
                status, landmarks = facemark.fit(frame, faces)
                if args.verbose:
                    print("status : ", status)
                    print("landmarks : ", landmarks)
                    print("time taken to fit : {0:.2f} ms".format(1000*(time.time() - start)))
                for f in range(len(landmarks)):
                    cv2.face.drawFacemarks(frame, landmarks[f])
            else:
                print("No Face Detected")

            cv2.imshow('Landmark_Window', frame)
            if cv2.waitKey(0) == 27:
              break;
        else:
            break
    cv2.destroyAllWindows()
