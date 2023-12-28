# YOLOv8 Pose Trained on MS-COCO Data
___

**Microsoft COCO is a pose detection project that uses the latest version of YOLO (You Only Look Once) Version 8, YOLO models developed by Ultralytics in the dataset used for image recognition, segmentation, captioning, object detection and keypoint estimation consisting of more than three hundred thousand images.**

## Model Information

![Model_Information](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/model_info.png)

```mathematica
PacletInstall["NeuralNetworks"]
```
![PacletObject](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/PacletObject.png)

### Resource Retrieval
___
Get the pre-trained net:

```mathematica
NetModel["YOLO V8 Pose Trained on MS-COCO Data"]
```
![NetModel](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/NetModel.gif)

### NetModel Parameters
___
This model consists of a family of individual nets, each identified by a specific parameter combination. Inspect the available parameters:

```mathematica
NetModel["YOLO V8 Pose Trained on MS-COCO Data", \
"ParametersInformation"]
```

![ParametersInformation](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/ParametersInformation.png)

A non-default net by specifying the parameters:

```mathematica
NetModel[{"YOLO V8 Pose Trained on MS-COCO Data", "Size" -> "X"}]
```
![NetGraphX](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/NetGraphX.gif)

A non-default uninitialized net:

```mathematica
NetModel[{"YOLO V8 Pose Trained on MS-COCO Data", 
  "Size" -> "X"}, "UninitializedEvaluationNet"]
```
![UninitializedEvaluationNet](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/UninitializedEvaluationNet.gif)

### Evaluation Function
___
```mathematica
labels = {"Nose", "LeftEye", "RightEye", "LeftEar", "RightEar", 
   "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow", 
   "LeftWrist", "RightWrist", "LeftHip", "RightHip", "LeftKnee", 
   "RightKnee", "LeftAnkle", "RightAnkle"};
```

```mathematica
netevaluate[net_, img_, detectionThreshold_ : .25, 
   overlapThreshold_ : .5] := Module[
   {imgSize, w, h, probableObj, probableBoxes, probableScores, 
    probableKeypoints, max, scale, padx, pady, results, nms, x1, y1, 
    x2, y2},
   
   (*define image dimensions*)
   imgSize = 640;
   {w, h} = ImageDimensions[img];
   
   (*get inference*)
   results = net[img];
   
   (*filter by probability*)
   (*very small probability are thresholded*)
   probableObj = 
    UnitStep[
     results["Objectness"] - detectionThreshold]; {probableBoxes, 
     probableScores, probableKeypoints} = 
    Map[Pick[#, probableObj, 1] &, {results["Boxes"], 
      results["Objectness"], results["KeyPoints"]}];
   If[Or[Length[probableBoxes] == 0, Length[probableKeypoints] == 0], 
    Return[{}]];
   
   max = Max[{w, h}];
   scale = max/imgSize;
   {padx, pady} = imgSize*(1 - {w, h}/max)/2;
   
   (*transform keypoints coordinates to fit the input image size*)
   probableKeypoints = Apply[
     {
       {
        Clip[Floor[scale*(#1 - padx)], {1, w}],
        Clip[Floor[scale*(imgSize - #2 - pady)], {1, h}]
        },
       #3
       } &,
     probableKeypoints, {2}
     ];
   
   (*transform coordinates into rectangular boxes*)
   probableBoxes = Apply[
     (
       x1 = Clip[Floor[scale*(#1 - #3/2 - padx)], {1, w}];
       y1 = Clip[Floor[scale*(imgSize - #2 - #4/2 - pady)], {1, h}];
       x2 = Clip[Floor[scale*(#1 + #3/2 - padx)], {1, w}];
       y2 = Clip[Floor[scale*(imgSize - #2 + #4/2 - pady)], {1, h}];
       Rectangle[{x1, y1}, {x2, y2}]
       ) &, probableBoxes, 1
     ];
   
   (*gather the boxes of the same class and perform non-
   max suppression*)
   nms = ResourceFunction["NonMaximumSuppression"][
     probableBoxes -> probableScores, "Index", 
     MaxOverlapFraction -> overlapThreshold];
   results = Association[];
   results["ObjectDetection"] = 
    Part[Transpose[{probableBoxes, probableScores}], nms];
   results["KeypointEstimation"] = 
    Part[probableKeypoints, nms][[All, All, 1]];
   results["KeypointConfidence"] = 
    Part[probableKeypoints, nms][[All, All, 2]];
   results
   ];
```
### Model Usage
___

Obtain the detected bounding boxes with their corresponding classes and confidences as well as the locations of human joints for a given image:

```mathematica
testImage = ![testImage](testImage.jpg);
```

```
predictions = 
  netevaluate[NetModel["YOLO V8 Pose Trained on MS-COCO Data"], 
   testImage];
```

Inspect the prediction keys:

```mathematica
Keys[predictions]
```

```
{"ObjectDetection", "KeypointEstimation", "KeypointConfidence"}
```

The "ObjectDetection" key contains the coordinates of the detected objects as well as their confidences and classes:

```mathematica
predictions["ObjectDetection"]
```

```
{{Rectangle[{97, 45}, {414, 578}], 0.868033}}
```

The "KeypointEstimation" key contains the locations of the top predicted keypoints:

```mathematica
predictions["KeypointEstimation"]
```

```
{{{224, 506}, {240, 520}, {210, 519}, {263, 506}, {188, 505}, {294, 
   451}, {162, 441}, {338, 370}, {150, 349}, {292, 301}, {216, 
   308}, {283, 317}, {184, 311}, {345, 332}, {160, 324}, {357, 
   137}, {169, 131}}}
```

The "KeypointConfidence" key contains the confidences for each person’s keypoints:

```mathematica
predictions["KeypointConfidence"]
```

```
{{0.996321, 0.985294, 0.98312, 0.7473, 0.656033, 0.99429, 0.991576, 
  0.989896, 0.986335, 0.993605, 0.992221, 0.995908, 0.995443, 
  0.994574, 0.994403, 0.963511, 0.963552}}
```

```mathematica
keypoints = predictions["KeypointEstimation"];
```

Visualize the keypoints:

```mathematica
HighlightImage[testImage, keypoints]
```
![HighlightImage](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/HighlightImage.jpg)

Visualize the keypoints grouped by person:

```mathematica
HighlightImage[testImage, 
 AssociationThread[Range[Length[keypoints]] -> keypoints], 
 ImageLabels -> None]
```
![HighlightImage1](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/HighlightImage1.jpg)

Visualize the keypoints grouped by a keypoint type:

```mathematica
HighlightImage[testImage, 
 AssociationThread[
  Range[Length[Transpose@keypoints]] -> Transpose@keypoints], 
 ImageLabels -> None, ImageLegends -> labels]
```
![HighlightImage2](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/HighlightImage2.jpg)

```mathematica
getSkeleton[personKeypoints_] := 
 Line[DeleteMissing[
   Map[personKeypoints[[#]] &, {{1, 2}, {1, 3}, {2, 4}, {3, 5}, {1, 
    6}, {1, 7}, {6, 8}, {8, 10}, {7, 9}, {9, 11}, {6, 7}, {6, 12}, {7,
     13}, {12, 13}, {12, 14}, {14, 16}, {13, 15}, {15, 17}}], 1, 2]]
```

Visualize the pose keypoints, object detections and human skeletons:

```mathematica
HighlightImage[testImage,
 AssociationThread[Range[Length[#]] -> #] & /@ {keypoints, 
   Map[getSkeleton, keypoints], 
   predictions["ObjectDetection"][[;; , 1]]},
 ImageLabels -> None
 ]
```
![HighlightImage3](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/HighlightImage3.jpg)

### Network Result
___

The network computes eight thousand four hundred bounding boxes, the position of the keypoints with their probabilities and the probability of an object inside the box:

```mathematica
res = NetModel["YOLO V8 Pose Trained on MS-COCO Data"][testImage];
```

```mathematica
Dimensions /@ res
```

```
<|"Boxes" -> {8400, 4}, "KeyPoints" -> {8400, 17, 3}, 
 "Objectness" -> {8400}|>
```

Rescale the "KeyPoints" to the coordinates of the input image and visualize them scaled and colored by their probability measures:

```mathematica
imgSize = 640;
{w, h} = ImageDimensions[testImage];
max = Max[{w, h}];
scale = max/imgSize;
{padx, pady} = imgSize*(1 - {w, h}/max)/2;
heatpoints = Flatten[Apply[
    {
      {Clip[Floor[scale*(#1 - padx)], {1, w}],
        Clip[Floor[scale*(imgSize - #2 - pady)], {1, h}]
        } ->
       ColorData["TemperatureMap"][#3]
      } &,
    res["KeyPoints"], {2}
    ]];
```

```mathematica
heatmap = ReplaceImageValue[ConstantImage[1, {w, h}], heatpoints]
```
![heatmap](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/heatmap.jpg)

Overlay the heat map on the image:

```mathematica
ImageCompose[testImage, {heatmap, 0.6}]
```
![ImageCompose](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/ImageCompose.jpg)

Rescale the bounding boxes to the coordinates of the input image and visualize them scaled by their "Objectness" measures:

```mathematica
boxes = Apply[
   (
     x1 = Clip[Floor[scale*(#1 - #3/2 - padx)], {1, w}];
     y1 = Clip[Floor[scale*(imgSize - #2 - #4/2 - pady)], {1, h}];
     x2 = Clip[Floor[scale*(#1 + #3/2 - padx)], {1, w}];
     y2 = Clip[Floor[scale*(imgSize - #2 + #4/2 - pady)], {1, h}];
     Rectangle[{x1, y1}, {x2, y2}]
     ) &, res["Boxes"], 1
   ];
```

```mathematica
Graphics[
 MapThread[{EdgeForm[Opacity[Total[#1] + .01]], #2} &, {res[
    "Objectness"], boxes}], 
 BaseStyle -> {FaceForm[], EdgeForm[{Thin, Black}]}]
```
![Graphics](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/Graphics.jpg)

Superimpose the predictions on top of the input received by the net:

```mathematica
HighlightImage[testImage, 
 Graphics[
  MapThread[{EdgeForm[Opacity[Total[#1] + .01]], #2} &, {res[
     "Objectness"], boxes}], 
  BaseStyle -> {FaceForm[], EdgeForm[{Thin, Black}]}], 
 BaseStyle -> {FaceForm[], EdgeForm[{Thin, Red}]}]
```
![HighlightImage4](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/HighlightImage4.jpg)

### Net Information
___

Inspect the number of parameters of all arrays in the net:

```mathematica
Information[
 NetModel[
  "YOLO V8 Pose Trained on MS-COCO Data"], "ArraysElementCounts"]
```

```
<|{"Bn_1", "Biases"} -> 16, {"Bn_1", "MovingMean"} -> 
  16, {"Bn_1", "MovingVariance"} -> 16, {"Bn_1", "Scaling"} -> 
  16, {"Bn_2", "Biases"} -> 32, {"Bn_2", "MovingMean"} -> 
  32, {"Bn_2", "MovingVariance"} -> 32, {"Bn_2", "Scaling"} -> 
  32, {"Bn_3", "Biases"} -> 64, {"Bn_3", "MovingMean"} -> 
  64, {"Bn_3", "MovingVariance"} -> 64, {"Bn_3", "Scaling"} -> 
  64, {"Bn_4", "Biases"} -> 128, {"Bn_4", "MovingMean"} -> 
  128, {"Bn_4", "MovingVariance"} -> 128, {"Bn_4", "Scaling"} -> 
  128, {"Bn_5", "Biases"} -> 256, {"Bn_5", "MovingMean"} -> 
  256, {"Bn_5", "MovingVariance"} -> 256, {"Bn_5", "Scaling"} -> 
  256, {"Bn_6", "Biases"} -> 64, {"Bn_6", "MovingMean"} -> 
  64, {"Bn_6", "MovingVariance"} -> 64, {"Bn_6", "Scaling"} -> 
  64, {"Bn_7", "Biases"} -> 128, {"Bn_7", "MovingMean"} -> 
  128, {"Bn_7", "MovingVariance"} -> 128, {"Bn_7", "Scaling"} -> 
  128, {"Conv_1", "Weights"} -> 432, {"Conv_2", "Weights"} -> 
  4608, {"Conv_3", "Weights"} -> 18432, {"Conv_4", "Weights"} -> 
  73728, {"Conv_5", "Weights"} -> 294912, {"Conv_6", "Weights"} -> 
  36864, {"Conv_7", "Weights"} -> 
  147456, {"C2f_1", "Bn_1", "Biases"} -> 
  64, {"C2f_1", "Bn_1", "MovingMean"} -> 
  64, {"C2f_1", "Bn_1", "MovingVariance"} -> 
  64, {"C2f_1", "Bn_1", "Scaling"} -> 
  64, {"C2f_1", "Bn_2", "Biases"} -> 
  64, {"C2f_1", "Bn_2", "MovingMean"} -> 
  64, {"C2f_1", "Bn_2", "MovingVariance"} -> 
  64, {"C2f_1", "Bn_2", "Scaling"} -> 
  64, {"C2f_1", "Conv_1", "Weights"} -> 
  4096, {"C2f_1", "Conv_2", "Weights"} -> 
  8192, {"C2f_2", "Bn_1", "Biases"} -> 
  128, {"C2f_2", "Bn_1", "MovingMean"} -> 
  128, {"C2f_2", "Bn_1", "MovingVariance"} -> 
  128, {"C2f_2", "Bn_1", "Scaling"} -> 
  128, {"C2f_2", "Bn_2", "Biases"} -> 
  128, {"C2f_2", "Bn_2", "MovingMean"} -> 
  128, {"C2f_2", "Bn_2", "MovingVariance"} -> 
  128, {"C2f_2", "Bn_2", "Scaling"} -> 
  128, {"C2f_2", "Conv_1", "Weights"} -> 
  16384, {"C2f_2", "Conv_2", "Weights"} -> 
  32768, {"C2f_3", "Bn_1", "Biases"} -> 
  128, {"C2f_3", "Bn_1", "MovingMean"} -> 
  128, {"C2f_3", "Bn_1", "MovingVariance"} -> 
  128, {"C2f_3", "Bn_1", "Scaling"} -> 
  128, {"C2f_3", "Bn_2", "Biases"} -> 
  128, {"C2f_3", "Bn_2", "MovingMean"} -> 
  128, {"C2f_3", "Bn_2", "MovingVariance"} -> 
  128, {"C2f_3", "Bn_2", "Scaling"} -> 
  128, {"C2f_3", "Conv_1", "Weights"} -> 
  49152, {"C2f_3", "Conv_2", "Weights"} -> 
  24576, {"C2f_4", "Bn_1", "Biases"} -> 
  64, {"C2f_4", "Bn_1", "MovingMean"} -> 
  64, {"C2f_4", "Bn_1", "MovingVariance"} -> 
  64, {"C2f_4", "Bn_1", "Scaling"} -> 
  64, {"C2f_4", "Bn_2", "Biases"} -> 
  64, {"C2f_4", "Bn_2", "MovingMean"} -> 
  64, {"C2f_4", "Bn_2", "MovingVariance"} -> 
  64, {"C2f_4", "Bn_2", "Scaling"} -> 
  64, {"C2f_4", "Conv_1", "Weights"} -> 
  12288, {"C2f_4", "Conv_2", "Weights"} -> 
  6144, {"C2f_5", "Bn_1", "Biases"} -> 
  128, {"C2f_5", "Bn_1", "MovingMean"} -> 
  128, {"C2f_5", "Bn_1", "MovingVariance"} -> 
  128, {"C2f_5", "Bn_1", "Scaling"} -> 
  128, {"C2f_5", "Bn_2", "Biases"} -> 
  128, {"C2f_5", "Bn_2", "MovingMean"} -> 
  128, {"C2f_5", "Bn_2", "MovingVariance"} -> 
  128, {"C2f_5", "Bn_2", "Scaling"} -> 
  128, {"C2f_5", "Conv_1", "Weights"} -> 
  24576, {"C2f_5", "Conv_2", "Weights"} -> 
  24576, {"C2f_6", "Bn_1", "Biases"} -> 
  256, {"C2f_6", "Bn_1", "MovingMean"} -> 
  256, {"C2f_6", "Bn_1", "MovingVariance"} -> 
  256, {"C2f_6", "Bn_1", "Scaling"} -> 
  256, {"C2f_6", "Bn_2", "Biases"} -> 
  256, {"C2f_6", "Bn_2", "MovingMean"} -> 
  256, {"C2f_6", "Bn_2", "MovingVariance"} -> 
  256, {"C2f_6", "Bn_2", "Scaling"} -> 
  256, {"C2f_6", "Conv_1", "Weights"} -> 
  98304, {"C2f_6", "Conv_2", "Weights"} -> 
  98304, {"C2f_7", "Bn_1", "Biases"} -> 
  32, {"C2f_7", "Bn_1", "MovingMean"} -> 
  32, {"C2f_7", "Bn_1", "MovingVariance"} -> 
  32, {"C2f_7", "Bn_1", "Scaling"} -> 
  32, {"C2f_7", "Bn_2", "Biases"} -> 
  32, {"C2f_7", "Bn_2", "MovingMean"} -> 
  32, {"C2f_7", "Bn_2", "MovingVariance"} -> 
  32, {"C2f_7", "Bn_2", "Scaling"} -> 
  32, {"C2f_7", "Conv_1", "Weights"} -> 
  1024, {"C2f_7", "Conv_2", "Weights"} -> 
  1536, {"C2f_8", "Bn_1", "Biases"} -> 
  256, {"C2f_8", "Bn_1", "MovingMean"} -> 
  256, {"C2f_8", "Bn_1", "MovingVariance"} -> 
  256, {"C2f_8", "Bn_1", "Scaling"} -> 
  256, {"C2f_8", "Bn_2", "Biases"} -> 
  256, {"C2f_8", "Bn_2", "MovingMean"} -> 
  256, {"C2f_8", "Bn_2", "MovingVariance"} -> 
  256, {"C2f_8", "Bn_2", "Scaling"} -> 
  256, {"C2f_8", "Conv_1", "Weights"} -> 
  65536, {"C2f_8", "Conv_2", "Weights"} -> 
  98304, {"Detect", "Constant_1", "Array"} -> 
  16800, {"Detect", "Constant_2", "Array"} -> 
  1, {"Detect", "Constant_3", "Array"} -> 
  8400, {"Detect", "Constant_4", "Array"} -> 
  16800, {"SPPF", "Bn_1", "Biases"} -> 
  128, {"SPPF", "Bn_1", "MovingMean"} -> 
  128, {"SPPF", "Bn_1", "MovingVariance"} -> 
  128, {"SPPF", "Bn_1", "Scaling"} -> 
  128, {"SPPF", "Bn_2", "Biases"} -> 
  256, {"SPPF", "Bn_2", "MovingMean"} -> 
  256, {"SPPF", "Bn_2", "MovingVariance"} -> 
  256, {"SPPF", "Bn_2", "Scaling"} -> 
  256, {"SPPF", "Conv_1", "Weights"} -> 
  32768, {"SPPF", "Conv_2", "Weights"} -> 
  131072, {"C2f_1", "Bottleneck_1", "Bn_1", "Biases"} -> 
  32, {"C2f_1", "Bottleneck_1", "Bn_1", "MovingMean"} -> 
  32, {"C2f_1", "Bottleneck_1", "Bn_1", "MovingVariance"} -> 
  32, {"C2f_1", "Bottleneck_1", "Bn_1", "Scaling"} -> 
  32, {"C2f_1", "Bottleneck_1", "Bn_2", "Biases"} -> 
  32, {"C2f_1", "Bottleneck_1", "Bn_2", "MovingMean"} -> 
  32, {"C2f_1", "Bottleneck_1", "Bn_2", "MovingVariance"} -> 
  32, {"C2f_1", "Bottleneck_1", "Bn_2", "Scaling"} -> 
  32, {"C2f_1", "Bottleneck_1", "Conv_1", "Weights"} -> 
  9216, {"C2f_1", "Bottleneck_1", "Conv_2", "Weights"} -> 
  9216, {"C2f_1", "Bottleneck_2", "Bn_1", "Biases"} -> 
  32, {"C2f_1", "Bottleneck_2", "Bn_1", "MovingMean"} -> 
  32, {"C2f_1", "Bottleneck_2", "Bn_1", "MovingVariance"} -> 
  32, {"C2f_1", "Bottleneck_2", "Bn_1", "Scaling"} -> 
  32, {"C2f_1", "Bottleneck_2", "Bn_2", "Biases"} -> 
  32, {"C2f_1", "Bottleneck_2", "Bn_2", "MovingMean"} -> 
  32, {"C2f_1", "Bottleneck_2", "Bn_2", "MovingVariance"} -> 
  32, {"C2f_1", "Bottleneck_2", "Bn_2", "Scaling"} -> 
  32, {"C2f_1", "Bottleneck_2", "Conv_1", "Weights"} -> 
  9216, {"C2f_1", "Bottleneck_2", "Conv_2", "Weights"} -> 
  9216, {"C2f_2", "Bottleneck_1", "Bn_1", "Biases"} -> 
  64, {"C2f_2", "Bottleneck_1", "Bn_1", "MovingMean"} -> 
  64, {"C2f_2", "Bottleneck_1", "Bn_1", "MovingVariance"} -> 
  64, {"C2f_2", "Bottleneck_1", "Bn_1", "Scaling"} -> 
  64, {"C2f_2", "Bottleneck_1", "Bn_2", "Biases"} -> 
  64, {"C2f_2", "Bottleneck_1", "Bn_2", "MovingMean"} -> 
  64, {"C2f_2", "Bottleneck_1", "Bn_2", "MovingVariance"} -> 
  64, {"C2f_2", "Bottleneck_1", "Bn_2", "Scaling"} -> 
  64, {"C2f_2", "Bottleneck_1", "Conv_1", "Weights"} -> 
  36864, {"C2f_2", "Bottleneck_1", "Conv_2", "Weights"} -> 
  36864, {"C2f_2", "Bottleneck_2", "Bn_1", "Biases"} -> 
  64, {"C2f_2", "Bottleneck_2", "Bn_1", "MovingMean"} -> 
  64, {"C2f_2", "Bottleneck_2", "Bn_1", "MovingVariance"} -> 
  64, {"C2f_2", "Bottleneck_2", "Bn_1", "Scaling"} -> 
  64, {"C2f_2", "Bottleneck_2", "Bn_2", "Biases"} -> 
  64, {"C2f_2", "Bottleneck_2", "Bn_2", "MovingMean"} -> 
  64, {"C2f_2", "Bottleneck_2", "Bn_2", "MovingVariance"} -> 
  64, {"C2f_2", "Bottleneck_2", "Bn_2", "Scaling"} -> 
  64, {"C2f_2", "Bottleneck_2", "Conv_1", "Weights"} -> 
  36864, {"C2f_2", "Bottleneck_2", "Conv_2", "Weights"} -> 
  36864, {"C2f_3", "Bottleneck", "Bn_1", "Biases"} -> 
  64, {"C2f_3", "Bottleneck", "Bn_1", "MovingMean"} -> 
  64, {"C2f_3", "Bottleneck", "Bn_1", "MovingVariance"} -> 
  64, {"C2f_3", "Bottleneck", "Bn_1", "Scaling"} -> 
  64, {"C2f_3", "Bottleneck", "Bn_2", "Biases"} -> 
  64, {"C2f_3", "Bottleneck", "Bn_2", "MovingMean"} -> 
  64, {"C2f_3", "Bottleneck", "Bn_2", "MovingVariance"} -> 
  64, {"C2f_3", "Bottleneck", "Bn_2", "Scaling"} -> 
  64, {"C2f_3", "Bottleneck", "Conv_1", "Weights"} -> 
  36864, {"C2f_3", "Bottleneck", "Conv_2", "Weights"} -> 
  36864, {"C2f_4", "Bottleneck", "Bn_1", "Biases"} -> 
  32, {"C2f_4", "Bottleneck", "Bn_1", "MovingMean"} -> 
  32, {"C2f_4", "Bottleneck", "Bn_1", "MovingVariance"} -> 
  32, {"C2f_4", "Bottleneck", "Bn_1", "Scaling"} -> 
  32, {"C2f_4", "Bottleneck", "Bn_2", "Biases"} -> 
  32, {"C2f_4", "Bottleneck", "Bn_2", "MovingMean"} -> 
  32, {"C2f_4", "Bottleneck", "Bn_2", "MovingVariance"} -> 
  32, {"C2f_4", "Bottleneck", "Bn_2", "Scaling"} -> 
  32, {"C2f_4", "Bottleneck", "Conv_1", "Weights"} -> 
  9216, {"C2f_4", "Bottleneck", "Conv_2", "Weights"} -> 
  9216, {"C2f_5", "Bottleneck", "Bn_1", "Biases"} -> 
  64, {"C2f_5", "Bottleneck", "Bn_1", "MovingMean"} -> 
  64, {"C2f_5", "Bottleneck", "Bn_1", "MovingVariance"} -> 
  64, {"C2f_5", "Bottleneck", "Bn_1", "Scaling"} -> 
  64, {"C2f_5", "Bottleneck", "Bn_2", "Biases"} -> 
  64, {"C2f_5", "Bottleneck", "Bn_2", "MovingMean"} -> 
  64, {"C2f_5", "Bottleneck", "Bn_2", "MovingVariance"} -> 
  64, {"C2f_5", "Bottleneck", "Bn_2", "Scaling"} -> 
  64, {"C2f_5", "Bottleneck", "Conv_1", "Weights"} -> 
  36864, {"C2f_5", "Bottleneck", "Conv_2", "Weights"} -> 
  36864, {"C2f_6", "Bottleneck", "Bn_1", "Biases"} -> 
  128, {"C2f_6", "Bottleneck", "Bn_1", "MovingMean"} -> 
  128, {"C2f_6", "Bottleneck", "Bn_1", "MovingVariance"} -> 
  128, {"C2f_6", "Bottleneck", "Bn_1", "Scaling"} -> 
  128, {"C2f_6", "Bottleneck", "Bn_2", "Biases"} -> 
  128, {"C2f_6", "Bottleneck", "Bn_2", "MovingMean"} -> 
  128, {"C2f_6", "Bottleneck", "Bn_2", "MovingVariance"} -> 
  128, {"C2f_6", "Bottleneck", "Bn_2", "Scaling"} -> 
  128, {"C2f_6", "Bottleneck", "Conv_1", "Weights"} -> 
  147456, {"C2f_6", "Bottleneck", "Conv_2", "Weights"} -> 
  147456, {"C2f_7", "Bottleneck", "Bn_1", "Biases"} -> 
  16, {"C2f_7", "Bottleneck", "Bn_1", "MovingMean"} -> 
  16, {"C2f_7", "Bottleneck", "Bn_1", "MovingVariance"} -> 
  16, {"C2f_7", "Bottleneck", "Bn_1", "Scaling"} -> 
  16, {"C2f_7", "Bottleneck", "Bn_2", "Biases"} -> 
  16, {"C2f_7", "Bottleneck", "Bn_2", "MovingMean"} -> 
  16, {"C2f_7", "Bottleneck", "Bn_2", "MovingVariance"} -> 
  16, {"C2f_7", "Bottleneck", "Bn_2", "Scaling"} -> 
  16, {"C2f_7", "Bottleneck", "Conv_1", "Weights"} -> 
  2304, {"C2f_7", "Bottleneck", "Conv_2", "Weights"} -> 
  2304, {"C2f_8", "Bottleneck", "Bn_1", "Biases"} -> 
  128, {"C2f_8", "Bottleneck", "Bn_1", "MovingMean"} -> 
  128, {"C2f_8", "Bottleneck", "Bn_1", "MovingVariance"} -> 
  128, {"C2f_8", "Bottleneck", "Bn_1", "Scaling"} -> 
  128, {"C2f_8", "Bottleneck", "Bn_2", "Biases"} -> 
  128, {"C2f_8", "Bottleneck", "Bn_2", "MovingMean"} -> 
  128, {"C2f_8", "Bottleneck", "Bn_2", "MovingVariance"} -> 
  128, {"C2f_8", "Bottleneck", "Bn_2", "Scaling"} -> 
  128, {"C2f_8", "Bottleneck", "Conv_1", "Weights"} -> 
  147456, {"C2f_8", "Bottleneck", "Conv_2", "Weights"} -> 
  147456, {"Detect", "cv1", "Bn_1", "Biases"} -> 
  64, {"Detect", "cv1", "Bn_1", "MovingMean"} -> 
  64, {"Detect", "cv1", "Bn_1", "MovingVariance"} -> 
  64, {"Detect", "cv1", "Bn_1", "Scaling"} -> 
  64, {"Detect", "cv1", "Bn_2", "Biases"} -> 
  64, {"Detect", "cv1", "Bn_2", "MovingMean"} -> 
  64, {"Detect", "cv1", "Bn_2", "MovingVariance"} -> 
  64, {"Detect", "cv1", "Bn_2", "Scaling"} -> 
  64, {"Detect", "cv1", "Conv", "Biases"} -> 
  64, {"Detect", "cv1", "Conv", "Weights"} -> 
  4096, {"Detect", "cv1", "Conv_1", "Weights"} -> 
  36864, {"Detect", "cv1", "Conv_2", "Weights"} -> 
  36864, {"Detect", "cv2", "Bn_1", "Biases"} -> 
  64, {"Detect", "cv2", "Bn_1", "MovingMean"} -> 
  64, {"Detect", "cv2", "Bn_1", "MovingVariance"} -> 
  64, {"Detect", "cv2", "Bn_1", "Scaling"} -> 
  64, {"Detect", "cv2", "Bn_2", "Biases"} -> 
  64, {"Detect", "cv2", "Bn_2", "MovingMean"} -> 
  64, {"Detect", "cv2", "Bn_2", "MovingVariance"} -> 
  64, {"Detect", "cv2", "Bn_2", "Scaling"} -> 
  64, {"Detect", "cv2", "Conv", "Biases"} -> 
  64, {"Detect", "cv2", "Conv", "Weights"} -> 
  4096, {"Detect", "cv2", "Conv_1", "Weights"} -> 
  73728, {"Detect", "cv2", "Conv_2", "Weights"} -> 
  36864, {"Detect", "cv3", "Bn_1", "Biases"} -> 
  64, {"Detect", "cv3", "Bn_1", "MovingMean"} -> 
  64, {"Detect", "cv3", "Bn_1", "MovingVariance"} -> 
  64, {"Detect", "cv3", "Bn_1", "Scaling"} -> 
  64, {"Detect", "cv3", "Bn_2", "Biases"} -> 
  64, {"Detect", "cv3", "Bn_2", "MovingMean"} -> 
  64, {"Detect", "cv3", "Bn_2", "MovingVariance"} -> 
  64, {"Detect", "cv3", "Bn_2", "Scaling"} -> 
  64, {"Detect", "cv3", "Conv", "Biases"} -> 
  64, {"Detect", "cv3", "Conv", "Weights"} -> 
  4096, {"Detect", "cv3", "Conv_1", "Weights"} -> 
  147456, {"Detect", "cv3", "Conv_2", "Weights"} -> 
  36864, {"Detect", "cv4", "Bn_1", "Biases"} -> 
  64, {"Detect", "cv4", "Bn_1", "MovingMean"} -> 
  64, {"Detect", "cv4", "Bn_1", "MovingVariance"} -> 
  64, {"Detect", "cv4", "Bn_1", "Scaling"} -> 
  64, {"Detect", "cv4", "Bn_2", "Biases"} -> 
  64, {"Detect", "cv4", "Bn_2", "MovingMean"} -> 
  64, {"Detect", "cv4", "Bn_2", "MovingVariance"} -> 
  64, {"Detect", "cv4", "Bn_2", "Scaling"} -> 
  64, {"Detect", "cv4", "Conv", "Biases"} -> 
  1, {"Detect", "cv4", "Conv", "Weights"} -> 
  64, {"Detect", "cv4", "Conv_1", "Weights"} -> 
  36864, {"Detect", "cv4", "Conv_2", "Weights"} -> 
  36864, {"Detect", "cv5", "Bn_1", "Biases"} -> 
  64, {"Detect", "cv5", "Bn_1", "MovingMean"} -> 
  64, {"Detect", "cv5", "Bn_1", "MovingVariance"} -> 
  64, {"Detect", "cv5", "Bn_1", "Scaling"} -> 
  64, {"Detect", "cv5", "Bn_2", "Biases"} -> 
  64, {"Detect", "cv5", "Bn_2", "MovingMean"} -> 
  64, {"Detect", "cv5", "Bn_2", "MovingVariance"} -> 
  64, {"Detect", "cv5", "Bn_2", "Scaling"} -> 
  64, {"Detect", "cv5", "Conv", "Biases"} -> 
  1, {"Detect", "cv5", "Conv", "Weights"} -> 
  64, {"Detect", "cv5", "Conv_1", "Weights"} -> 
  73728, {"Detect", "cv5", "Conv_2", "Weights"} -> 
  36864, {"Detect", "cv6", "Bn_1", "Biases"} -> 
  64, {"Detect", "cv6", "Bn_1", "MovingMean"} -> 
  64, {"Detect", "cv6", "Bn_1", "MovingVariance"} -> 
  64, {"Detect", "cv6", "Bn_1", "Scaling"} -> 
  64, {"Detect", "cv6", "Bn_2", "Biases"} -> 
  64, {"Detect", "cv6", "Bn_2", "MovingMean"} -> 
  64, {"Detect", "cv6", "Bn_2", "MovingVariance"} -> 
  64, {"Detect", "cv6", "Bn_2", "Scaling"} -> 
  64, {"Detect", "cv6", "Conv", "Biases"} -> 
  1, {"Detect", "cv6", "Conv", "Weights"} -> 
  64, {"Detect", "cv6", "Conv_1", "Weights"} -> 
  147456, {"Detect", "cv6", "Conv_2", "Weights"} -> 
  36864, {"Detect", "cv7", "Bn_1", "Biases"} -> 
  51, {"Detect", "cv7", "Bn_1", "MovingMean"} -> 
  51, {"Detect", "cv7", "Bn_1", "MovingVariance"} -> 
  51, {"Detect", "cv7", "Bn_1", "Scaling"} -> 
  51, {"Detect", "cv7", "Bn_2", "Biases"} -> 
  51, {"Detect", "cv7", "Bn_2", "MovingMean"} -> 
  51, {"Detect", "cv7", "Bn_2", "MovingVariance"} -> 
  51, {"Detect", "cv7", "Bn_2", "Scaling"} -> 
  51, {"Detect", "cv7", "Conv", "Biases"} -> 
  51, {"Detect", "cv7", "Conv", "Weights"} -> 
  2601, {"Detect", "cv7", "Conv_1", "Weights"} -> 
  29376, {"Detect", "cv7", "Conv_2", "Weights"} -> 
  23409, {"Detect", "cv8", "Bn_1", "Biases"} -> 
  51, {"Detect", "cv8", "Bn_1", "MovingMean"} -> 
  51, {"Detect", "cv8", "Bn_1", "MovingVariance"} -> 
  51, {"Detect", "cv8", "Bn_1", "Scaling"} -> 
  51, {"Detect", "cv8", "Bn_2", "Biases"} -> 
  51, {"Detect", "cv8", "Bn_2", "MovingMean"} -> 
  51, {"Detect", "cv8", "Bn_2", "MovingVariance"} -> 
  51, {"Detect", "cv8", "Bn_2", "Scaling"} -> 
  51, {"Detect", "cv8", "Conv", "Biases"} -> 
  51, {"Detect", "cv8", "Conv", "Weights"} -> 
  2601, {"Detect", "cv8", "Conv_1", "Weights"} -> 
  58752, {"Detect", "cv8", "Conv_2", "Weights"} -> 
  23409, {"Detect", "cv9", "Bn_1", "Biases"} -> 
  51, {"Detect", "cv9", "Bn_1", "MovingMean"} -> 
  51, {"Detect", "cv9", "Bn_1", "MovingVariance"} -> 
  51, {"Detect", "cv9", "Bn_1", "Scaling"} -> 
  51, {"Detect", "cv9", "Bn_2", "Biases"} -> 
  51, {"Detect", "cv9", "Bn_2", "MovingMean"} -> 
  51, {"Detect", "cv9", "Bn_2", "MovingVariance"} -> 
  51, {"Detect", "cv9", "Bn_2", "Scaling"} -> 
  51, {"Detect", "cv9", "Conv", "Biases"} -> 
  51, {"Detect", "cv9", "Conv", "Weights"} -> 
  2601, {"Detect", "cv9", "Conv_1", "Weights"} -> 
  117504, {"Detect", "cv9", "Conv_2", "Weights"} -> 
  23409, {"Detect", "dfl", "Conv", "Weights"} -> 16|>
```

Obtain the total number of parameters:

```mathematica
Information[
 NetModel[
  "YOLO V8 Pose Trained on MS-COCO Data"], "ArraysTotalElementCount"]
```

```
3348483
```

Obtain the layer type counts:

```mathematica
Information[
 NetModel["YOLO V8 Pose Trained on MS-COCO Data"], "LayerTypeCounts"]
```

```
<|ConvolutionLayer -> 73, BatchNormalizationLayer -> 63, 
 ElementwiseLayer -> 65, PartLayer -> 17, ThreadingLayer -> 15, 
 CatenateLayer -> 21, PoolingLayer -> 3, ResizeLayer -> 2, 
 NetArrayLayer -> 4, ReshapeLayer -> 11, TransposeLayer -> 3, 
 SoftmaxLayer -> 1|>
```

Display the summary graphic:

```mathematica
Information[
 NetModel["YOLO V8 Pose Trained on MS-COCO Data"], "SummaryGraphic"]
```
![Information](https://github.com/kemalkilicaslan/YOLOv8_Pose_Trained_on_MS-COCO_Data/blob/main/Information.jpg)
