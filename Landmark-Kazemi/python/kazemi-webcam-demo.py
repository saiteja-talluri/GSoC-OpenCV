import os
import time
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstration of kazemi facial landmark algorithm.')
    parser.add_argument('--face_cascade', type=str, help="Path to the cascade model file for the face detector",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','lbpcascade_frontalface_improved.xml'))
    parser.add_argument('--kazemi_model', type=str, help="Path to the kazemi trained model file",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','face_landmark_model.dat'))
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--path_to_save_video',type=str, help='Output path of video of demonstration.', default='output/face_landmarks.avi')
    parser.add_argument('--width', type=int, help='width of the window', default=960)
    parser.add_argument('--height', type=int, help='height of the window', default=720)
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    video_capture = None

    if args.from_video:
        assert os.path.isfile(args.from_video)
        video_capture = cv2.VideoCapture(args.from_video)
    else:
        video_capture = cv2.VideoCapture(args.camera_id)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    face_cascade = cv2.CascadeClassifier(args.face_cascade)
    facemark = cv2.face.createFacemarkKazemi()
    facemark.loadModel(args.kazemi_model)

    cv2.namedWindow('Landmark_Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Landmark_Window', args.width, args.height)

    if args.record_video:
        record_video = cv2.VideoWriter(args.path_to_save_video, cv2.VideoWriter_fourcc('M','J','P','G'), 10,
                                        (int( video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int( video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while(video_capture.isOpened()):
        ret, frame = video_capture.read()
        if ret :
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
                if args.verbose:
                    print("No Face Detected")
            if args.record_video:
                record_video.write(frame)
            cv2.imshow('Landmark_Window', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    video_capture.release()
    if args.record_video:
        record_video.release()
    cv2.destroyAllWindows()
