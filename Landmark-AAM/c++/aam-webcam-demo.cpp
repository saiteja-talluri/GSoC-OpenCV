/*----------------------------------------------
 * Usage:
 * aam-webcam-demo <face_cascade_model> <eyes_cascade_model> <aam_model>
 *
 * Example:
 * aam-webcam-demo models/face_cascade.xml models/eyes_cascade.xml models/AAM.yaml
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

    vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

bool getInitialFitting(Mat image, Rect face, vector<Point2f> s0, CascadeClassifier eyes_cascade, Mat & R, Point2f & Trans, float & scale){
    vector<Point2f> mybase;
    vector<Point2f> T;
    vector<Point2f> base = Mat(Mat(s0)+Scalar(image.cols/2,image.rows/2)).reshape(2);

    vector<Point2f> base_shape,base_shape2 ;
    Point2f e1 = Point2f((float)((base[39].x+base[36].x)/2.0),(float)((base[39].y+base[36].y)/2.0)); //eye1
    Point2f e2 = Point2f((float)((base[45].x+base[42].x)/2.0),(float)((base[45].y+base[42].y)/2.0)); //eye2

    if(face.width==0 || face.height==0) return false;

    vector<Point2f> eye;
    bool found=false;

    Mat faceROI = image( face);
    vector<Rect> eyes;

    // In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, CASCADE_SCALE_IMAGE, Size(20, 20) );
    if(eyes.size()==2){
        found = true;
        int j=0;
        Point2f c1( (float)(face.x + eyes[j].x + eyes[j].width*0.5), (float)(face.y + eyes[j].y + eyes[j].height*0.5));

        j=1;
        Point2f c2( (float)(face.x + eyes[j].x + eyes[j].width*0.5), (float)(face.y + eyes[j].y + eyes[j].height*0.5));

        Point2f pivot;
        double a0,a1;
        if(c1.x<c2.x){
            pivot = c1;
            a0 = atan2(c2.y-c1.y, c2.x-c1.x);
        }
        else{
            pivot = c2;
            a0 = atan2(c1.y-c2.y, c1.x-c2.x);
        }

        scale = (float)(norm(Mat(c1)-Mat(c2))/norm(Mat(e1)-Mat(e2)));

        mybase= Mat(Mat(s0)*scale).reshape(2);
        Point2f ey1 = Point2f((float)((mybase[39].x+mybase[36].x)/2.0),(float)((mybase[39].y+mybase[36].y)/2.0));
        Point2f ey2 = Point2f((float)((mybase[45].x+mybase[42].x)/2.0),(float)((mybase[45].y+mybase[42].y)/2.0));


        #define TO_DEGREE 180.0/3.14159265
        a1 = atan2(ey2.y-ey1.y, ey2.x-ey1.x);
        Mat rot = getRotationMatrix2D(Point2f(0,0), (a1-a0)*TO_DEGREE, 1.0);

        rot(Rect(0,0,2,2)).convertTo(R, CV_32F);

        base_shape = Mat(Mat(R*scale*Mat(Mat(s0).reshape(1)).t()).t()).reshape(2);
        ey1 = Point2f((float)((base_shape[39].x+base_shape[36].x)/2.0),(float)((base_shape[39].y+base_shape[36].y)/2.0));
        ey2 = Point2f((float)((base_shape[45].x+base_shape[42].x)/2.0),(float)((base_shape[45].y+base_shape[42].y)/2.0));

        T.push_back(Point2f(pivot.x-ey1.x,pivot.y-ey1.y));
        Trans = Point2f(pivot.x-ey1.x,pivot.y-ey1.y);
        return true;
    }
    else{
        Trans = Point2f( (float)(face.x + face.width*0.5),(float)(face.y + face.height*0.5));
    }
    return found;
}

bool parseArguments(int argc, char** argv,
    String & cascade,
    String & model,
    String & aam_model
){
    const String keys =
        "{ @f face-cascade    |      | (required) path to the cascade model file for the face detector }"
        "{ @e eyes-cascade    |      | (required) path to the cascade model file for the eyes detector }"
        "{ @m model-path      |      | (required) path of the AAM model to be loaded}"
        "{ help h usage ?     |      | aam-webcam-demo -face-cascade -eyes-cascade -model \n"
             " example: aam-webcam-demo models/face_cascade.xml models/eyes_cascade.xml models/AAM.yaml }"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("hello");

    if (parser.has("help")){
        parser.printMessage();
        return false;
    }

    cascade = String(parser.get<string>("face-cascade"));
    model = String(parser.get<string>("eyes-cascade"));
    aam_model = String(parser.get<string>("model-path"));

    if(cascade.empty() || model.empty() || aam_model.empty()){
        cerr << "one or more required arguments are not found" << '\n';
        cout<<"face-cascade : "<<cascade.c_str()<<endl;
        cout<<"eyes-cascade : "<<model.c_str()<<endl;
        cout<<"aam-model : "<<aam_model.c_str()<<endl;
        parser.printMessage();
        return false;
    }
    return true;
}


int main(int argc, char** argv )
{
    String cascade_path, eyes_cascade_path, aam_model;
    if(!parseArguments(argc, argv, cascade_path, eyes_cascade_path, aam_model))
       return -1;


    /* Create the Facemark Instance */
    FacemarkAAM::Params params;
    params.scales.push_back(2.0);
    params.scales.push_back(4.0);
    params.model_filename = aam_model;
    Ptr<FacemarkAAM> facemark = FacemarkAAM::create(params);
    facemark->loadModel(aam_model);

    /* Trainsformation Variables */
    float scale ;
    Point2f T;
    Mat R;

    /* Set up webcam for video capture */
    VideoCapture cam(0);


    /* Initialise the video writer object */
    VideoWriter video("webcam_landmarks.avi",VideoWriter::fourcc('M','J','P','G'),10, Size(cam.get(cv::CAP_PROP_FRAME_WIDTH),cam.get(cv::CAP_PROP_FRAME_HEIGHT)),true);
    if ( !video.isOpened() ) //if VideoWriter did not initliazr successfully, exit the program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }

    /* Base Shape */
    Mat frame, image;
    FacemarkAAM::Data data;
    facemark->getData(&data);
    vector<Point2f> s0 = data.s0;

    /* Load the Cascade classifiers */
    vector<Rect> faces;
    CascadeClassifier face_cascade(cascade_path);
    CascadeClassifier eyes_cascade(eyes_cascade_path);

    while(cam.read(image)){
        // Detect Faces
        myDetector(image, faces, &face_cascade);

        if(faces.size() > 0){
            // Get Initialization
            vector<FacemarkAAM::Config> conf;
            vector<Rect> faces_eyes;
            for(unsigned j=0; j<faces.size(); j++){
                if(getInitialFitting(image,faces[j],s0,eyes_cascade, R,T,scale)){
                    conf.push_back(FacemarkAAM::Config(R,T,scale,(int)params.scales.size()-1));
                    faces_eyes.push_back(faces[j]);
                }
            }

            // Fitting Process
            if(conf.size()> 0){
                vector<vector<Point2f> > landmarks;
                facemark->fitConfig(image, faces_eyes, landmarks, conf);
                for(unsigned j=0; j<landmarks.size();j++){
                    drawFacemarks(image, landmarks[j],Scalar(0,255,0));
                }
            }
        }
        imshow("Landmark Image", image);
        video.write(image);
        if (waitKey(1) == 27)
          break;
    }
}
