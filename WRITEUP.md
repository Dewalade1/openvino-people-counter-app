# Project Write-Up

The people counter application is a smart video IoT solution that can detect people in a designated area of observation, providing the number of people in the frame, average duration of people in frame, total count of people since the start of the observation session and an alarm that sends an alert to the UI telling the user when a person enters the video frame. It alerts the user when the total count of people that have entered the video since the start of the observation session is greater than five. It was developed as a project required to graduate the Udacity and Intel AI at the Edge Nanodegree program.

The app makes use of Intel® hardware and software tools for deployment. The people counter app makes use of the Inference Engine included in the Intel® Distribution of openVINO™ Toolkit to run it's Edge computations, inferencing and classification processes. The model results are then filtered by a python script (the main.py file) to only identify people in a video recording, camera feed or image.

The app will count the number of people in the current frame by counting the number of times a bounding box is drawn in that frame. It will also calculate the duration of time spent by a person in the frame (time elapsed between entering and exiting a frame). This is calculated by subtracting the time the person exits the video from the time the same person entered the video. It also counts the total count of people since the start of the video stream till the streaming ends or is interrupted. It then sends the data to a local web server using the Paho MQTT Python package. The app also saves a copy of the streamed output to the local storage device. The streaming video can be viewed live by connecting to the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) using a web browser.

## Summary of Results

Upon testing the people counter app with videos and images of objects besides people, for example cars, dogs e.t.c., it was discovered that the app did not draw any bounding boxes for images other than those classified as people. All other object classes were ignored except the 'person' class.

In conclusion, the app works as intended and identifies only people present in the video or image. It also performs all calculations and inferences as intended and returns the desired results in the desired formats through the UI accessed via the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) using a web browser or in the local storage at the `outputs\videos\` for video outputs and `outputs\images\` directory for images. A few videos and images have been provided in the `resources` folder located in the home directory to use to the app.

For more details, please refer to the full write-up below.

## Model Info

In my investigation of potential models for the people counter app, I found a suitable model that works well for the purpose of our app. Below is info about the model:

Model: `faster_rcnn_inception_v2_coco_2018_01_28`

Framework: `Tensorflow`

Total size: `216MB`

Device type used for conversion: `CPU`

Contents of folder: `model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`, `pipeline.config`, `frozen_inference_graph.pb`, `frozen_inference_graph.mapping`,`checkpoint` and a folder named `saved_model`(which contains the `saved_model.pb` file and an empty folder named `variables`)

Download link: `http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`

link to Paper:

- <https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf>

- <https://arxiv.org/pdf/1506.01497.pdf>

Brief Description:
Faster RCNN is an object detection model presented by Ross Girshick, Shaoqing Ren, Kaiming He and Jian Sun in 2015, and is one of the famous object detection architectures that uses convolution neural networks. Fast R-CNN passes the entire image to a ConvNet which generates regions of interest instead of passing the extracted regions from the image like a regular R-CNN. Instead of using three different models LIKE the R-CNN, it uses a single model which extracts features from the regions, classifies them into different classes, and returns the bounding boxes. All these steps are done simultaneously, thus making it execute faster than the R-CNN. Fast R-CNN is, however, not fast enough when applied on a large dataset as it also uses selective search for extracting the regions.

Faster R-CNN fixes the problem by replacing it with the Region Proposal Network (RPN). It would first extract the feature maps from the input image using a ConvNet and then pass them to a RPN which returns object proposals then these maps are classified and the bounding boxes are predicted.

More info about the model can be found [here](https://towardsdatascience.com/faster-r-cnn-for-object-detection-a-technical-summary-474c5b857b46)

**Note:** This brief description is an excerpt from the analyticsvidhya blog post titled [A Practical Implementation of the Faster R-CNN Algorithm for Object Detection (Part 2 – with Python codes)](https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/).

### CPU INFO

The model was converted to an Intermediate Representation and the app was run on a local device with the following specifications:

Name: `Intel(R) Core(TM) i5-825OU @1.60GHz`

Clock Rate: `1800MHz`

RAM: `16GB`

Operating System: `Windows 10`

IR generation execution time: `167.69seconds`

IR Model precision: `FP32`. This was chosen to ensure maximum accuracy in the model.

### Acquiring and converting the model

The model was downloaded as a zipped file from the tensorflow model zoo and then unzipped from the terminal using the following commands:

```bash
 wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
 tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

After unzipping, I changed directory to the folder containing the model files using the command below:

```bash
 cd faster_rcnn_inception_v2_coco_2018_01_28 && ls -a
```

The model was then converted to an Intermediate Representation with the use of the Model Optimizer using the following command:

```bash
 python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```

After these, the generated Intermediate Representation of the model (Its generated XML file and BIN file) were moved to a new folder named `model` and the folder containing the pre-conversion models files were deleted running the following commands at the app's home directory:

```bash
rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
rm -r faster_rcnn_inception_v2_coco_2018_01_28
```

After this, the people counter app was developed using the Intermediate representation of the model.

**Note:** These commands only work on Linux and Mac terminals. They will not work on an unconfigured windows command prompt.

## Explaining Custom Layers

### What are custom layers

Rarely, there are some layers in a neural network model that are not in the openVINO Model Optimizer supported layers list. They are layers that are not natively supported by the openvino Inference engine. These layers can be added to the Inference Engine as custom layers. The custom layer can therefore be defined as any model layer that is not natively supported by the model framework.

Though the custom layer is a useful feature and there can be some unsupported layers in our model. The custom layer feature of the model optimizer is rarely needed because these unsupported layers are usually supported by the built-in device extensions in the openvino tool kit. For example, the unsupported layers in the faster rcnn model are the 'proposals', 'Squeeze_3' and 'detection_output' layers but all these layers are supported by the CPU extension and GPU extension available in the openVINO toolkit. Thus, no custom layers were needed in order to use this model with the openVINO Inference Engine. Rather, we just need to add the right extension to the Inference engine core.

### How it works

The process behind converting custom layers involves two necessary custom layer extensions:

1. Custom Layer Extractor
2. Custom Layer Operation

### 1. Custom Layer Extractor

This extension is responsible for identifying and extracting the parameters for each instance of custom layers. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output Intermediate Representation. The input layer parameters are usually left unchanged during conversion to a custom layer as shown in the tutorial.

### 2. Custom Layer Operation

The custom layer operation specifies the attributes that are supported by the custom layer and computes the output shape for each instance of the custom layer from its parameters.

### Why we need custom layers

Some of the potential reasons for handling custom layers are:

- when there are layers in the model that are not supported by the model optimizer and also not supported by the available built-in device extensions. This is, however, a rare case.

- to allow the model optimizer to fully convert a model to an Intermediate Representation.

## Comparing Model Performance

My method to compare models before and after conversion to Intermediate Representations was to run an experiment. This experiment involved testing both versions of the model on the same set of videos comparing the recorded results of both trials. The model was converted to Intermediate Representation with an accuracy of F64 so as to get the highest possible accuracy from the model. However, this also means that the size of the Intermediate representation will be the largest possible size. In my comparison of both versions of the models, I discovered the following:

### Accuracy

The pre-conversion model had an accuracy of 94.55% throughout the duration of the streaming test. The probability threshold used was 60% and as a result, no bounding boxes were drawn for confidence levels below 60%. This resulted in some of the frames that had a person in them having no bounding boxes drawn. This is because the model must have predicted with a less than 60% probability that there was a person in the frame, causing the frame to have a blinking effect during video streaming. This has minimal effects on the results as it happened few times during the streaming test. But overall, the pre-conversion model made predictions wit good accuracies.

The accuracy of the model at post-conversion to Intermediate Representation averaged at 94.67% through out the duration of the testing of the video stream. The probability threshold used was 60% and as a result, no bounding boxes were drawn for confidence levels below 60%. This resulted in some of the frames that had a person in them having no bounding boxes drawn around those persons. This indicates that the model must have predicted with a less than 60% probability that there was a person in the frame, causing the frame to have a blinking effect during video streaming. However, this happened few times during the streaming of the sample video. But overall, the post-conversion model made predictions with very good accuracy.

The post-conversion model was just as accurate as the pre-conversion model. This is most likely because the model was converted to Intermediate Representation with a precision of FP32 bits. This is the highest possible accuracy of an Intermediate Representation. It resulted in no accuracy loss during model conversion.

### Size

The total size of the pre-conversion model after unzipping was 166MB. The model protobuff file had a size of 54.5MB, its checkpoint files (model.ckpt.data-00000-of-00001, model.ckpt.index and model.ckpt.meta) had a total size of 56.3 MB. This is quite large if we wish to use it for applications at the Edge. It is not very suited for Edge applications as it raises costs for our Edge application's storage device thus raising our overall budget for the Edge application.

For the model Intermediate Representation, the XML file size was 124 KB and the BIN file size was 50.7MB. This brings the total size of the post-conversion model (XML file and BIN file) to 50.8MB. This is a very small file size and doesn't take up much storage space at all. It saves a lot of costs on storage device making our budget for Edge applications cheaper.

From the above, the post conversion model size is significantly smaller than the pre-conversion model. The post-conversion model is 30.6% of the total size of the pre-conversion model and 90% of the protobuff file of the pre-conversion model. The post conversion model size is affected by the precision of the model Intermediate Representation. Our model was set to a precision of FP32 bits during conversion. This results in a trade-off where we get the largest possible model size but also get the highest accuracy possible from our model. The post conversion model could have been made smaller if the model was converted to a lower precision but we risk loosing accuracy. The small size of the Intermediate Representation files makes it suitable for Edge applications.

### Inference time

When the pre-conversion model was tested on the classroom workspace, The minimum inference time of the pre-conversion model was 928.126ms and its maximum inference time was 942.889ms. This resulted an average inference time of 935.5075ms. After testing the pre-conversion model on a local device, the results were a lot different. The minimum inference time of the pre-conversion model reduced to 90.564ms and the maximum time of the model reduced to 94.254ms. This

The minimum inference time of the post-conversion model was 881.151ms and its maximum inference time was 889.344ms This resulted an average inference time of 885.2475ms when the model was tested on the workspace provided. However, when tested local device, the minimum inference time was 83.157ms and the maximum inference time was 85.254ms. Thus, locally, the average inference time was 84.205ms.

Judging from the large difference in the averages of the local and online tests, The high inference time on the workspace test was deduced to be as a result of poor network connection on my end. However, looking at the average inference time of the app when tested locally, it was very good. We can say that our app will still perform well when deployed on devices with low processing power.

### CPU Overhead

CPU overhead of the pre-conversion model was around 70% per core. This is an acceptable overhead for a CPU as it is within the accepted limit of CPU overhead if we intend to make use of the app for long periods of time without breaks. It doesn't overclock the CPU and wouldn't cause any damages to it although it is likely to cause the CPU to overheat if the model is deployed for extensive periods of time.

CPU overhead of the post-conversion model was around 40%. This is a very good overhead for a CPU as it makes use of as little CPU processing power as possible. If we intend to make use of the app for long periods of time without breaks, this would be the most ideal one to use. It doesn't take up much CPU resources and wouldn't cause any overheating or damages to the CPU if the model is deployed for extensive periods of time on said CPU.

The Intermediate Representation model is obviously better at minimizing CPU Overhead than the raw model itself. This is because the Intermediate Representation of a model is designed to optimize its processes thus making use of as little processing power as possible. This frees up processing power for other secondary applications or processes that may be intended for the Edge application.

### Differences between deploying at the edge and using cloud services

When deploying at the Edge, the app requires low amounts of internet speed to exchange data with the server. This is because all processing and calculation is done on the local device and only the result is sent to the server or recipient device though the internet or edge model can be used with very low internet speeds compared to deploying on the cloud.

Cost of renting quality servers for data storage and processing is high. Cloud services perform majority of their computations on servers and thus for a good cloud-based app, quality servers are needed. Also, the larger the scope of our app, the larger the amount of computations hence, the greater the number of servers needed to perform these computations as quickly as possible. This results in an exponential increase in hardware costs. This cost is saved when deploying at the edge as the device only needs minimal processing power to run. This is because the app runs locally on a device and requires little to no servers to operate it.

Edge computing also results in lower latency than cloud services. This is because cloud services are highly dependent on the internet for processing which usually takes place on a server in a different location from the location of the application device itself. This device's functionality can be compromised if it's deployed in remote areas where there is little internet speed. The Edge applications, however, do not have this problem because all computing is done locally and only the results are sent to the recipients' devices. This greatly minimizes network requirements and also reduces latency to the bare minimum. Remember that the average inference time of our app on the classroom workspace was 885.2475ms and the average inference time was 84.205ms when tested on my local device. If we take the online workspace as a form of cloud processing, and the local device as Edge processing, we can say that Edge computing results lower latency than cloud computing. This means that better service can be provided when deploying our app in remote areas by deploying at the edge than if we relied on cloud computing.

For the reasons stated above, when deploying in remote areas, Edge applications are more suitable than cloud services. This is because edge applications rely on as little network connection and hardware resources as possible for operation and as a result save costs on resource unlike the cloud services which rely heavily on these resources and would therefore require more to perform optimally.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

1. Monitoring of civilian movements within public places such as parks, banks, theme parks, cinemas and so on during the period of the COVID-19 lockdown.

2. Monitoring the number of people within a room to prevent overcrowding by setting an alarm that goes off when the limit on the number of people in the room at a time has been exceeded.

3. To detect intruders in restricted areas or private properties. The app does this by raising an alarm when it detects a person within the camera's range of vision.

4. In retail, control of the people and the time inside the shop could be important in order to select the exposition and other issues.

5. It can be used on drones to be deployed by first responders to cover a lot of ground in a short period of time, reducing response time and in turn increase the rate of success of search and rescue missions conducted in large areas.

6. Crime detection in city-wide areas through the use of already installed cameras within the city's perimeter.

7. Searching city-wide or nation-wide areas for wanted felons using cameras in a city combined with the people counter app equipped with a model trained to detect that individual felon or a group of felons.

8. For monitoring worker behaviors and mass-movement trends of workers in a factory, mine or production facility where man-power is utilized.

9. When combined with pattern detection software, the people counter app can be used to detect patterns in a person's movements and predict the likelihood of that person being a potential robber based on their movement patterns.

10. When combined with armed defense systems, it can be used to track, locate and deter intruders of highly restricted areas and locations.

Each of these use cases is be possible because the people counter app deployed at the edge is capable of sending the desired information to the user's mobile device in real-time irrespective of their location with little to no latency. This is made possible by the low network requirements of the app combined with its fast processing speeds. Due to its low hardware requirements, setup costs of edge applications are significantly lower than most of its counterparts, such as cloud computing, making it possible to deploy the application in large scale projects such as some of the ones mentioned above on a relatively low budget.

## Assess Effects on End User Needs

Lighting, model accuracy, weather, visibility, camera focal length, and image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

- **Lighting:** The lighting in the camera's area of observation is the most important in order to obtain a good result. If the lighting in the image is poor or the area being monitored is dark, the model would not be able to detect any person within the video frame. The lighting can be affected by many factors.

  In the outdoors, lighting is primarily affected by the time of day, amount of artificial light sources (like street lamps) in the area, amount of lit sources of fires and animals. In the indoors, lighting is primarily sourced artificially. Some places also use glow-in-the-dark plants and animals and fires as indoor sources of light.

  Thus, monitored areas must have good lighting. If good lighting cannot be provided, a good alternative would be to make use of night vision cameras to counter low lighting situations. Although this approach is more costly than using a regular camera and the model may have to be retrained to detect people in images produced by night vision cameras, it is very effective at eliminating problems relating to lighting.

- **Model accuracy:** Deployed edge models must be able to make highly accurate detections. This is because these models results are usually required for real time usage and if they are deployed with models that have poor accuracy, it would make the app a lot less reliable. This is because if incorrect classifications made by the model was fed to the Edge application. It could result in Edge application making incorrect decisions, based on those classification results, in situations and these decisions could be dangerous for end users or/and third parties such as property damage or loss of life. Thus the model must always only send results that have a high degree of accuracy to the app. Taking into account the amount of hinderances that come with deploying in remote areas, a minimum of 60% accuracy is recommended for models used in edge applications.

  The accuracy of a model can also be affected by external factors such as weather, visibility, device maintainance, camera focal length, and lighting. Each of these factors affect the model accuracy one way or another, some more severely than others depending on the location of deployment. Thus, in order to keep model accuracy as high as possible, all these factors must be taken into account and must be dealt with according to how much effect the have on the model accuracy.

- **Weather:** The performance of the people counter app is directly dependent on the visibility and lighting in the camera's area of view. A people counter app deployed in the outdoors may not perform optimally in limited visibility. This also holds true for the case of the weather. This is because weather is one of the factors that affects visibility in a region.

  In any weather that provides ample lighting and visibility, such as sunny or cloudy weather, the model is expected to perform at optimal capacity assuming all other factors are good. However, the performance of the model will most likely reduce in rainy, foggy, or hazy weather. It would generally be reduced in weather that causes a reduction in visibility and low lighting.

  The lifespan of hardware used by the Edge application is also directly dependent on the weather. This is because when exposed, the elements cause the physical components of the device to wear out much faster than when they are not exposed to the elements. To prevent this, Edge devices that are deployed in the outdoors are usually shielded from the elements by deploying them in element-resistant shells or containers made from special materials and also designed to resist the specific-elements plaguing the lifespan of the device in that area. The devices area also protected from the elements using special coating that is capable of resisting heat or cold depending on the weather of the area where the app is intended to be deployed.

- **Camera focal length:** A large focal length gives you more focus on specific object and a arrow-angled image, reducing the area of observation of the people counter app. This is most likely not what we want if we would be deploying the app in the outdoors or in a large room but it is good for monitoring specific locations like entrances and exits.

  Low focal length gives you a much wider area of observation and can be used to monitor a larger number of people at once. The images produced by low focal lengths are wide-angled making the low focal length camera well suited for surveillance. This focal length type is well suited for the outdoors or on surveillance drones, self-driving cars and other edge applications where a wide area would need to be monitored for the app to be useful.

  Both focal lengths are ideal for edge applications the only major differences are their areas of application and the types of information provided by the different focal lengths. High camera focal length provides more information about the person/people in the image but doesn't provide a lot of information on the time spent by that person in the area of observation, or the total number of people in the area at any single point in time because it focuses on a small subset of that area at any given time. On the other hand, the low focal length camera provides less information about each of the persons in the image and more about statistics such as the time spent by that person in the area of observation, the number of persons in said area, and so on.

- **Image size:** Image size determines the input size of the model. The image size depends on the resolution of the camera. A higher camera resolution results in a larger image file but also produces a more detailed image while a lesser camera resolution would result in a smaller image file with lesser details. A model is able to make more accurate classifications with a higher degree of accuracy if the resolution of the image is high but the trade-off is that this can also increase latency during video streaming and use up more storage space than we would want.

  The input image and output image can be resized to suite the end user needs and hardware requirements however one should note that reducing the image size could lead to loss information about the image contents which could reduce model accuracy during classification while increasing the image size does not necessarily lead to a better model accuracy. If the image was made too large, it could result in a blurry image because there would be too much of a difference between the old and new image sizes. This blurry image would be full of noise which, if left unfiltered, would reduce the accuracy of the model and in turn, the performance of the model.

  Thus, its best to use a model whose input size matches the resolution of the image produced by the camera. Some preprocessing can be done to filter out noise from the image but if any image resizing is going to be done, it should be done in moderation so as to retain or possibly improve model accuracy. The choice of image size and camera resolution should be made based on end user needs and resource availability in the area of deployment.

- **Visibility:** The visibility in the area of deployment directly affects the accuracy of models in the Edge application. If the app is deploys in areas with large amounts of particles in the air such as desserts, the visibility would be reduced on windy days and during sandstorms by sand particles carried by the wind. Snowstorms, hail, rain and hurricanes and gales also reduce the visibility in the area while bright and clear weather such as sunny or cloudy weather provide high levels of visibility. Other natural phenomena also affect the level of visibility in a cameras range of view.

  The visibility is also affected by the level of maintainance done on the camera. Assuming all other factors affecting visibility are good and the camera is well maintained and its lenses are cleaned regularly, the visibility would be good. However, a poorly maintained camera would have poor visibility because their would be a lot if dirt and residue on the camera lens. Poor visibility results in blurry or bad images being captured by the camera which affects the accuracy of the model's prediction. This in turn affects the performance of our application. Because of this, excellent visibility is recommended for optimal results. Good visibility can be achieved by regularly maintaining the camera, and especially, cleaning the lenses of the camera as often as possible. In situations where the visibility is affected by natural phenomena, the best that can be done would be to shield the devices and cameras from the elements so as to prolong their lifespan.

## Conclusion

Upon testing the people counter app with videos and images of objects besides people, for example cars, dogs etc, it was discovered that the app did not draw any bounding boxes for images other than those of people. All other object classes were ignored except the 'person' class. A few videos and images have been provided in the `resource` folder located in the home directory to use to confirm this.

In conclusion, the app works as intended and identifies only people present in the video or image. It also performs all calculations and inferences as intended and returns the desired results in the desired formats through the UI accessed via the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) using a web browser or in the local storage at the `outputs\videos\` for video outputs and `outputs\images\` directory for images.
