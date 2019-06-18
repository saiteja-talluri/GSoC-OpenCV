# GSoC-OpenCV
Code written for OpenCV during GSoC 2019 related to Facial Landmark Detection

<h2> Facial Landmark Detector </h2>
<ul>
  <li> <b>Description:</b> Facial feature detection and tracking is a high value area of computer vision since humans are interested in what humans are paying attention to, feeling, and enhancing face pictures in selfies etc. At this point, we feel this should just become a standard "built in" ability that computer vision users can just call/rely on. OpenCV already has some code available on for Facial Landmark detection, see <a href="https://docs.opencv.org/4.0.0/d5/d47/tutorial_table_of_content_facemark.html">Tutorial on Facial Landmark Detector API</a> and <a href="https://docs.opencv.org/4.0.0/de/d27/tutorial_table_of_content_face.html">Tutorials for face module</a>, but much progress has been made that we want to make available. The task is to create a Facial Landmark detector model with the following requirements:</li>
  
  <li><b>Expected Outcomes:</b>
    <ul>
      <li>For mobile, make available a smaller model with lesser points ( e.g. 5 instead of 68 )</li>
      <li>Run at better Accuracy than Dlib</li>
      <li>Face Stabilization ( filtering and optical flow )</li>
      <li>Add it to the Python API</li>
      <li>Resources</li>
      <ul>
        <li>https://docs.opencv.org/4.0.0/d5/d47/tutorial_table_of_content_facemark.html</li>
        <li>https://docs.opencv.org/4.0.0/de/d27/tutorial_table_of_content_face.html</li>
        <li>https://github.com/opencv/opencv_contrib/tree/master/modules/face</li>
      </ul>
    </ul>
  </li>
  <li><b> Skills Required: </b> Mastery level C++ and Python, college course work in Computer Vision that includes work on face features. Best if you have worked with deep networks before.</li>
  <li><b>Mentors:</b> Satya Mallick</li>
  <li><b>Difficulty:</b> Medium </li>
</ul>
