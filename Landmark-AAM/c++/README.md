# Active Appearance Model for Facial Landmark Detection in OpenCV
This repository contains code snippets to train and run a demo on webcam for detecting facial landmarks using the active apperance model in the OpenCV Facemark API.
 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

* Download the trained active appearance model from this [link](https://drive.google.com/open?id=1vE3sbc8vJY2M_s-BnRv-GGcXBEtd_gfI) and place the models in the [models](./models/) folder. 

* To run the webcam demo, go to the [build](./build) folder and run the following commands.
```
cmake ..
make
./aam-webcam-demo ../models/haarcascade_frontalface_alt2.xml ../models/haarcascade_eye_tree_eyeglasses.xml ../models/AAM.yaml
```
* To train the model, download the datasets available [here](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) and add the paths to the images and their corresponding landmark points in seperate text files. Refer [this](./train/) folder for the files used for training.
```
cmake ..
make
./aam-train ../models/haarcascade_frontalface_alt2.xml ../models/haarcascade_eye_tree_eyeglasses.xml ../models/my_AAM.yaml ../train/images_train.txt ../train/points_train.txt
```
Now you can find the trained model in the [models](./models/) folder. Note that for all the demo purposes the AAM.yaml model was used among all the models available in the [link](https://drive.google.com/open?id=1vE3sbc8vJY2M_s-BnRv-GGcXBEtd_gfI) provided above.

## Webcam Demo

<p align='center'>
  <img src='./output/webcam_landmarks.gif' alt='input'/>
</p>

## Future Work

*  Need to add python bindings to the OpenCV Facemark AAM and then implement this demo in python.
*  Need to train the model on more datasets, as of now I trained only on LFPW, AFW and IBUG due to memory limitations.

## Author

* Saiteja Talluri - [saiteja-talluri](https://github.com/saiteja-talluri)

## Acknowledgements

* I would like to thank [Laksono Kurnianggoro](https://github.com/kurnianggoro) for his wonderful implementation of AAM in the OpenCV Facemark API.
