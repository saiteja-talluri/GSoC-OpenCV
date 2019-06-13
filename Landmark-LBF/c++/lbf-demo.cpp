/*----------------------------------------------
 * Usage:
 * lbf-demo <face_cascade_model> <lbf_model> <testing_images>
 *
 * Example:
 * lbf-demo models/face_cascade.xml models/LBF.yaml test/images_test.txt
 *
 * Notes:
 * the user should provides the list of testing images in a text file
 * example of contents for images_test.txt:
 * ../trainset/image_0001.png
 * ../trainset/image_0002.png
 *--------------------------------------------------*/

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include "opencv2/core.hpp"
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
    String & lbf_model,
    String & test_images
){
    const String keys =
        "{ @f face-cascade    |      | (required) path to the cascade model file for the face detector }"
        "{ @m model-path      |      | (required) path to the trained LBF model}"
        "{ @t test-images     |      | (required) path of a text file contains the list of paths to the test images}"
        "{ help h usage ?     |      |  lbf-demo -face-cascade  -model -test_images \n"
             " example: lbf-demo models/face_cascade.xml  models/LBF.yaml test/images_test.txt }"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("hello");

    if (parser.has("help")){
        parser.printMessage();
        return false;
    }

    cascade = String(parser.get<string>("face-cascade"));
    lbf_model = String(parser.get<string>("model-path"));
    test_images = String(parser.get<string>("test-images"));

    if(cascade.empty() || lbf_model.empty() || test_images.empty()){
        cerr << "one or more required arguments are not found" << '\n';
        cout<<"face-cascade : "<<cascade.c_str()<<endl;
        cout<<"lbf-model : "<<lbf_model.c_str()<<endl;
        cout<<"test_images : "<<test_images.c_str()<<endl;
        parser.printMessage();
        return false;
    }
    return true;
}


int main(int argc, char** argv )
{
    String cascade_path, model_path, test_images_path;
    if(!parseArguments(argc, argv, cascade_path, model_path, test_images_path))
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

    /* Loading the testing dataset */
    String testFiles, testPts;
    testFiles = test_images_path;
    testPts = test_images_path; //unused
    vector<String> images;
    vector<String> facePoints;
    loadDatasetList(testFiles, testPts, images, facePoints);

    vector<Rect> faces;
    for(int i=0; i<(int)images.size(); i++){
        vector<vector<Point2f> > landmarks;

        Mat img = imread(images[i]);
        facemark->getFaces(img, faces);
        double newtime = (double)getTickCount();
        facemark->fit(img, faces, landmarks);
        double fittime = ((getTickCount() - newtime)/getTickFrequency());

        for(unsigned j=0;j<faces.size();j++){
            drawFacemarks(img, landmarks[j], Scalar(0,0,255));
            // rectangle(img, faces[j], Scalar(255,0,255));
        }

        if(faces.size() > 0){
            printf("Time taken - %f ms\n",fittime*1000);
            imshow("Landmark Image", img);;
            char k = waitKey(0);
            if (k == 27)
              break;
        }else{
            cout<<"Face not found in the Image"<<endl;
        }
    }
}