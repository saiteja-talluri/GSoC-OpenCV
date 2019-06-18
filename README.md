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

<h2> Updates </h2>
<ul>
  <li><b>[06/04/19]</b> Submitted the Proposal for the GSoC 2019 to OpenCV in Facial Landmark Detection. </li>
  <li><b>[07/05/19]</b> Received the acceptance letter for the GSoC 2019 in Facial Landmark Detection project.</li>
  <li><b>[27/06/19]</b> End of the Community Bonding Phase and Start of the Coding Phase. </li>
  <li><b>[02/06/19]</b> Completed training the AAM model and added links to the model and the sample C++ code has been provided.</li>
  <li><b>[07/06/19]</b> Trained the Kazemi model but the model seemes to not work as efficiently as the LBF and AAM models. May be more training is required but due to RAM limitations could not verify. </li>
  <li><b>[08/06/19]</b> Added sample C++ code for LBF model and LBF seems to dominate the AAM model in both speed and accuracy. </li>
  <li><b>[11/06/19]</b> Went through the implementation of the <a href="https://github.com/1adrianb/face-alignment">2D-and-3D face alignment</a> implementation and checked if the model can be added to openCV model zoo. <b>[Pushed to second priority as fixing the python bindings to the working LBF and AAM models is more important.]</b></li>
  <li><b>[15/06/19]</b> Went through all the issues and pull requests related to the python bindings on facemark API and these are few which need to be highlighted.
    <ul>
      <li>The discussion for the issue in the python bindings started with this <a href="https://github.com/opencv/opencv_contrib/issues/1661">issue</a> and many work arounds have been tried but many of them failed.</li>
      <li> However <a href="https://github.com/tegusi"> Gusi Te</a> proposed a method to fix the bindings and his <a href="https://github.com/opencv/opencv_contrib/pull/2069"> pull </a> request to opencv_contrib has been merged but his <a href="https://github.com/opencv/opencv/pull/14189"> pull </a> request to opencv failed as it was affecting a test file. This <a href="https://github.com/opencv/opencv_contrib/issues/14204"> issue </a> still persists as the pull to opencv failed.
      <li> The latest build of opencv_contrib has the same <a href="https://github.com/opencv/opencv_contrib/issues/2140">issue</a> in the fit function and is still open to solve.</li>
    </ul>
  </li>
<li><b>[17/06/19]</b> There also have been some failed attempts to expose the trainig code of the FaceMark API to the users in python. So I am working on these two things and hope it would get completed before the first evaluation. I was able to fix the fit fucntion but drawLandmarks gave a similar error (channels() == CV_MAT_CN(dtype)) may be of because of the fix, so I am looking into it.
  </li>
</ul>	

<h2> Planning and Schedule in Coding Phase </h2>

<table>
  <tbody>  
    <tr>
      <th>Date</th>
      <th>Progress</th>
      <th>Work Ahead</th>
    </tr>  	  
    <tr>
      <td> 27th May to 3rd June, 2019 </td>
      <td> 
        <ul>
          <li>Completed training the AAM model and added links to the model and the sample C++ code has been provided.</li>           </ul> 
      </td>
      <td>
        <ul>
          <li> Check the working of Kazemi Model and verify which of these models works the best, also have  a look at the Adrian bulats 3D Face Alignment implementation. </li>
        </ul>
      </td>
    </tr>        
    <tr>
      <td> 4th June to 11th June, 2019 </td>
      <td> 
        <ul>
          <li> Trained the Kazemi model but the model seemes to not work as efficiently as the LBF and AAM models. May be more training is required but due to RAM limitations could not verify. </li>
          <li> Added sample C++ code for LBF model and LBF seems to dominate the AAM model in both speed and accuracy. </li>
          <li> Went through the implementation of the <a href="https://github.com/1adrianb/face-alignment">2D-and-3D face alignment</a> implementation and checked if the model can be added to openCV model zoo. <b>[Pushed to second priority as fixing the python bindings to the working LBF and AAM models is more important.]</b></li>
        </ul>
      </td>
      <td>
        <ul>
          <li> Put the adrain bulats implementation as 2nd priority and start fixing the python bindings to the facemark API.           </li>
        </ul>
      </td>
    </tr>  
    <tr>
      <td> 12th June to 18th June, 2019 </td>
      <td> 
        <ul>
          <li> Went through all the issues and pull requests related to the python bindings on facemark API and these are few which need to be highlighted.
          <ul>
            <li>The discussion for the issue in the python bindings started with this <a href="https://github.com/opencv/opencv_contrib/issues/1661">issue</a> and many work arounds have been tried but many of them failed.</li>
            <li> However <a href="https://github.com/tegusi"> Gusi Te</a> proposed a method to fix the bindings and his <a href="https://github.com/opencv/opencv_contrib/pull/2069"> pull </a> request to opencv_contrib has been merged but his <a href="https://github.com/opencv/opencv/pull/14189"> pull </a> request to opencv failed as it was affecting a test file. This <a href="https://github.com/opencv/opencv_contrib/issues/14204"> issue </a> still persists as the pull to opencv failed.
            <li> The latest build of opencv_contrib has the same <a href="https://github.com/opencv/opencv_contrib/issues/2140">issue</a> in the fit function and is still open to solve.</li>
          </ul> 
          </li>
          <li> There also have been some failed attempts to expose the trainig code of the FaceMark API to the users in python. So I am working on these two things and hope it would get completed before the first evaluation. I was able to fix the fit fucntion but drawLandmarks gave a similar error (channels() == CV_MAT_CN(dtype)) may be of because of the fix, so I am looking into it.
        </li>
        </ul>
      </td>
      <td>
         <ul>
          <li> Fix the python bindings for the fit and the drawLandmarks functions in the facemark API </li>
          <li> Expose the training implementation of the Facemaek API to the python users and ensure this is done before the first evaluation. </li>
        </ul>
      </td>
    </tr>  
</tbody>
</table>
