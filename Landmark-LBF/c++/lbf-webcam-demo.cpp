/*----------------------------------------------
 * Usage:
 * lbf-webcam-demo <face_cascade_model> <lbf_model>
 *
 * Example:
 * lbf-webcam-demo models/face_cascade.xml models/LBF.yaml
 *
 *--------------------------------------------------*/

#include <stdio.h>
#include <fstream>
#include <sstream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

#include <iostream>
#include <string>
#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::face;

bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
    Mat gray;

    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();

    equalizeHist(gray, gray);

    std::vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

bool parseArguments(int argc, char** argv,
    String & cascade,
    String & lbf_model
){
    const String keys =
        "{ @f face-cascade    |      | (required) path to the cascade model file for the face detector }"
        "{ @m model-path      |      | (required) path to the trained LBF model}"
        "{ help h usage ?     |      |  lbf-demo -face-cascade  -model \n"
             " example: lbf-demo models/face_cascade.xml  models/LBF.yaml}"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("hello");

    if (parser.has("help")){
        parser.printMessage();
        return false;
    }

    cascade = String(parser.get<string>("face-cascade"));
    lbf_model = String(parser.get<string>("model-path"));

    if(cascade.empty() || lbf_model.empty()){
        cerr << "one or more required arguments are not found" << '\n';
        cout<<"face-cascade : "<<cascade.c_str()<<endl;
        cout<<"lbf-model : "<<lbf_model.c_str()<<endl;
        parser.printMessage();
        return false;
    }
    return true;
}


int main(int argc, char** argv )
{
    String cascade_path, model_path;
    if(!parseArguments(argc, argv, cascade_path, model_path))
       return -1;


    /* Create the Facemark Instance */
    FacemarkLBF::Params params;
    params.model_filename = model_path;
    params.cascade_face = cascade_path;
    Ptr<FacemarkLBF> facemark = FacemarkLBF::create(params);

    CascadeClassifier face_cascade;
    face_cascade.load(params.cascade_face.c_str());
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);
    facemark->loadModel(model_path);

    /* Set up webcam for video capture */
    VideoCapture cam(0);


    /* Initialise the video writer object */
    VideoWriter video("webcam_landmarks.avi",VideoWriter::fourcc('M','J','P','G'),10, Size(cam.get(cv::CAP_PROP_FRAME_WIDTH),cam.get(cv::CAP_PROP_FRAME_HEIGHT)),true);
    if ( !video.isOpened() ) //if VideoWriter did not initliazr successfully, exit the program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }

    Mat frame, img;
    String text;
    char buff[255];
    double fittime;
    int nfaces;
    vector<Rect> faces,faces_scaled;
    vector<vector<Point2f> > landmarks;
    namedWindow("Landmark Image", 1);

    while(cam.read(frame)){
        double __time__ = (double)getTickCount();
        float scale = (float)(400.0/frame.cols);
        resize(frame, img, Size((int)(frame.cols*scale), (int)(frame.rows*scale)), 0, 0, INTER_LINEAR_EXACT);
        facemark->getFaces(img,faces);
        faces_scaled.clear();

        for(int j=0;j<(int)faces.size();j++){
            faces_scaled.push_back(Rect(
                (int)(faces[j].x/scale),
                (int)(faces[j].y/scale),
                (int)(faces[j].width/scale),
                (int)(faces[j].height/scale)));
        }
        faces = faces_scaled;
        fittime=0;
        nfaces = (int)faces.size();
        if(faces.size() > 0){
            double newtime = (double)getTickCount();
            facemark->fit(frame, faces, landmarks);
            fittime = ((getTickCount() - newtime)/getTickFrequency());
            for(int j=0;j<(int)faces.size();j++){
                landmarks[j] = Mat(Mat(landmarks[j]));
                drawFacemarks(frame, landmarks[j], Scalar(0,0,255));
            }
        }
        double fps = (getTickFrequency()/(getTickCount() - __time__));
        sprintf(buff, "faces: %i %03.2f fps, fit:%03.0f ms",nfaces,fps,fittime*1000);
        text = buff;
        putText(frame, text, Point(20,40), FONT_HERSHEY_PLAIN , 2.0,Scalar::all(255), 2, 8);

        imshow("Landmark Image", frame);
        video.write(frame);
        if (waitKey(1) == 27)
          break;    
    }
}
