/*----------------------------------------------
 * Usage:
 * kazemi-train <face_cascade_model> <kazemi_model> <config_file> <training_images> <annotation_files> [test_files]
 *
 * Example:
 * kazemi-train models/face_cascade.xml models/facemark.dat models/config.xml trian/images_train.txt train/points_train.txt
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
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>
#include <vector>
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

bool parseArguments(int argc, char** argv,
    String & cascade,
    String & kazemi_model,
    String & config,
    String & images,
    String & annotations,
    Size & scale
){
    const String keys =
        "{ @f face_cascade  |      | (required) path to the face cascade xml file which you want to use as a detector. [example - models/face_cascade.xml] }"
        "{ @m model         |      | (required) path to model to be saved after training. [ example - models/model.dat] }"
        "{ @c config        |      | (required) path to configuration xml file containing parameters for training.[ example - models/config.xml] }"
        "{ @i images        |      | (required) path to images txt file      [example - train/images_train.txt] }"
        "{ @a annotations   |      | (required) path to annotations txt file [example - train/points_train.txt] }"
        "{ @w width         |  460 | The width which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ @h height        |  460 | The height which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ help h usage ?     |      | aam-train -images -annotations -config -model -face-cascade [-t]\n"
             " example: kazemi-train train/images_train.txt train/points_train.txt amodels/config.xml models/model.dat models/face_cascade.xml}"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("hello");

    if (parser.has("help")){
        parser.printMessage();
        return false;
    }

    images = String(parser.get<string>("images"));
    annotations = String(parser.get<string>("annotations"));
    config = String(parser.get<string>("config"));
    kazemi_model = String(parser.get<string>("model"));
    cascade = String(parser.get<string>("face_cascade"));
    scale = Size(parser.get<int>("width"),parser.get<int>("height"));

    if(cascade.empty() || kazemi_model.empty() || config.empty() || images.empty() || annotations.empty()){
        cerr <<"one or more required arguments are not found" << '\n';
        cout<<"images : "<<images.c_str()<<endl;
        cout<<"annotations : "<<annotations.c_str()<<endl;
        cout<<"config : "<<config.c_str()<<endl;
        cout<<"kazemi-model : "<<kazemi_model.c_str()<<endl;
        cout<<"face-cascade : "<<cascade.c_str()<<endl;
        
        parser.printMessage();
        return false;
    }
    return true;
}


int main(int argc, char** argv )
{
    String cascade_path, kazemi_model, config_file, images_path, annotations_path;
    Size scale(460,460);
    if(!parseArguments(argc, argv, cascade_path, kazemi_model, config_file, images_path, annotations_path, scale))
       return -1;

    /* Create the Facemark Instance */
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_path);
    FacemarkKazemi::Params params;
    params.configfile = config_file;
    params.faceCascadefile = cascade_path;
    params.modelfile = kazemi_model;
    params.scale = scale;
    Ptr<FacemarkKazemi> facemark = FacemarkKazemi::create(params);
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);

    
    vector<float> scale_(2,460);
    facemark->setParams(cascade_path, kazemi_model, config_file, scale_);
    

    /* Loading the training dataset */
    vector<String> images_train;
    vector<vector<Point2f> > landmarks_train;
    loadTrainingData(images_path, annotations_path, images_train, landmarks_train, 0.0);

    /* Adding the training samples */
    Mat image;
    // images_train.size()
    for(size_t i=0; i<10; i++){
        image = imread(images_train[i]);
        facemark->addTrainingSample(image, landmarks_train[i]);
    }
    cout << "Added the Training Data" <<endl;

    /* Start of training */
    /* Trained model will be saved to kazemi_model given */
    facemark->training();
}
