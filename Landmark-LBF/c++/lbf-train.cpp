/*----------------------------------------------
 * Usage:
 * lbf-train <face_cascade_model> <lbf_model> <training_images> <annotation_files> [test_files]
 *
 * Example:
 * lbf-train models/face_cascade.xml  models/AAM_trained.yaml trian/images_train.txt train/points_train.txt
 *
 * Notes:
 * the user should provides the list of training images_train
 * accompanied by their corresponding landmarks location in separated files.
 * example of contents for images_train.txt:
 * ../trainset/image_0001.png
 * ../trainset/image_0002.png
 * example of contents for points_train.txt:
 * ../trainset/image_0001.pts
 * ../trainset/image_0002.pts
 * where the image_xxxx.pts contains the position of each face landmark.
 * example of the contents:
 *  version: 1
 *  n_points:  68
 *  {
 *  115.167660 220.807529
 *  116.164839 245.721357
 *  120.208690 270.389841
 *  ...
 *  }
 * example of the dataset is available at https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip
 *--------------------------------------------------*/

#include <stdio.h>
#include <fstream>
#include <sstream>
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
    String & images,
    String & annotations
){
    const String keys =
        "{ @f face-cascade    |      | (required) path to the cascade model file for the face detector }"
        "{ @m model-path      |      | (required) path to save the trained LBF model}"
        "{ @i images          |      | (required) path of a text file contains the list of paths to all training images}"
        "{ @a annotations     |      | (required) path of a text file contains the list of paths to all training annotation files}"
        "{ help h usage ?     |      |  lbf-demo -face-cascade -model -images -annotations [-t]\n"
             " example: lbf-demo models/face_cascade.xml  models/LBF.yaml train/images_train.txt train/points_train.txt }"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("hello");

    if (parser.has("help")){
        parser.printMessage();
        return false;
    }

    cascade = String(parser.get<string>("face-cascade"));
    lbf_model = String(parser.get<string>("model-path"));
    images = String(parser.get<string>("images"));
    annotations = String(parser.get<string>("annotations"));

    if(cascade.empty() || lbf_model.empty() || images.empty() || annotations.empty()){
        cerr << "one or more required arguments are not found" << '\n';
        cout<<"face-cascade : "<<cascade.c_str()<<endl;
        cout<<"lbf-model : "<<lbf_model.c_str()<<endl;
        cout<<"images : "<<images.c_str()<<endl;
        cout<<"annotations : "<<annotations.c_str()<<endl;
        parser.printMessage();
        return false;
    }
    return true;
}

int main(int argc, char** argv )
{
    String cascade_path, model_path, images_path, annotations_path;
    if(!parseArguments(argc, argv, cascade_path, model_path, images_path, annotations_path))
       return -1;

    /* Create the Facemark Instance */
    FacemarkLBF::Params params;
    params.model_filename = model_path;
    params.cascade_face = cascade_path;
    Ptr<FacemarkLBF> facemark = FacemarkLBF::create(params);

    CascadeClassifier face_cascade;
    face_cascade.load(params.cascade_face.c_str());
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);

    /* Loading the training dataset */
    vector<String> images_train;
    vector<String> landmarks_train;
    loadDatasetList(images_path,annotations_path,images_train,landmarks_train);

    /* Adding the training samples */
    Mat image;
    vector<Point2f> facial_points;
    for(size_t i=0; i<images_train.size(); i++){
        image = imread(images_train[i].c_str());
        loadFacePoints(landmarks_train[i],facial_points);
        facemark->addTrainingSample(image, facial_points);
    }

    /* Start of training */
    /* Trained model will be saved to model_path given */
    facemark->training();
}
