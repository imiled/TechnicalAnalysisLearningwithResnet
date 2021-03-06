# TFM Introduction Deep Learning Tools For Finance 
## Application : Automated Financial Technical analysis using Resnet Neural Network

I focused in this project to apply a Transfert learning methodology of CNN to an image of the sp500 technical graph as an image. 

The objective is first to get a complete dataset, then a trained model based on Resnet infrastucture. 

In addition we analyse the model efficiency and last point for any image of a stock price historical graph we have a tool that tell us if we would rather buy, sell or hold.

I consider here a problem of behaviour finance as most investor look throughly at those graph more than fondamental numbers and those graph can be interpretated on small horizon (minutes) or long (years) to get an estimation of its evolution. The humain will process this information deeply and the consequence of this process is the behaviour of stock market. Benjamin Graham in the "Intelligent Investor" - written in 1949 and considered as the bible of value investing - introduce the allegory of Mr. Market, meant to personify the irrationality and group-think of the stock market. As of august 2020, the value of some stocks are higher than the economy of France and Germany and some companies (Tesla) are bought at a price quite difficult to apprehend in terms of valuation fondamental and comparison to established company in Europe (Volkswagen) or Japan (Toyota). That is true, that our brain is set to always find an explanation but in this approach we' ll try to apprehend the impact of price evolution to make Mr Market more greedy or fearful.

In this project I have chosen to present 4 steps which can be taken separately as we can load save datas or models. 

You can find more details in the ./docs/ repertory the TFM report and presentation. 

I worked on it in google colab you can see it in the following file  :

TechAnalysisLearningwithResnet.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imiled/TechnicalAnalysisLearningwithResnet/blob/main/TechnicalAnalysisLearningwithResnet.ipynb)

I recommend at start to use a virtual python envirenment using this specific command :
```
$ ./lib/init.sh
```

and using this Docker command to get the appropiate environment base:
```
$ docker pull imiled/sp500from_image_to_state_prediction:1.0
```

But it can be launched in local also using the command to get the specific packages either for a cpu or a gpu :

for cpu :
```
$ pip install -r cpu_requirements.txt
```

for a gpu :
```
$ pip install -r gpu_requirements.txt
```


Now for each step can be taken independently as we are saving loading datas and model at each time.

## Setup : Generate Dataset 

In this part we are generating the training and testing dataset and organise the directory

```
python3 lib/setup.py indexcodefromyahoo startyear endyear nbofblockforramoptim nbdayforpast nbofdayforfuture pathtoputtheimages
```

example

```
python3 lib/setup.py ^SPX 1927 2020 10 250 20 /content/TechnicalAnalysisLearningwithResnet/images/ 
```

indexcodefromyahoo can be for S&P 500 : ^SPX for ESTX 50 PR.EUR : ^STOXX50E and for FTSE 100 : ^FTSE
nbofblockforramoptim is a paremeter to optimize ram use as it cannot load all the images in the same app

First we download the historical prices of the sp500 from 1927 to 31 July 2020 and built the image of 250 days historical graph also we get the 20 days future price evolution of the sp500. 
From the future price evolution, we calculate a future state which can be splitted in 5 classes : ( an addtional ER ie error class is cleaned therafter)

**Sell-Sell | Sell- Neutral | Neutral | Neutral -Buy | Buy -Buy **

**SS | SN | NN | NB | BB **

![alt text](https://github.com/imiled/TechnicalAnalysisLearningwithResnet/blob/main/docs/explication201113.PNG)

The objective is to get a list of images representing the graph of the index during the past day for each date where the class of the image would be the evolution of the index in the next future days:


![alt text](https://github.com/imiled/TechnicalAnalysisLearningwithResnet/blob/main/docs/images%20dataset.PNG)


Please note that: 
1. I use cv2 and matplot lib to create the image in 255 x 255 x 1 for grayscale
2. I optimised the creation of the dataset using drive memory and not ram ( that would take all ram available and crash the system).

You can use the commands from *organisedataset.sh* to arrange the full images created in 2 dataset (training and testing) and for each class I randomly put the same number of images. Depending on the total number of images (min of class created) . For testing I have chosen 40 images per block so 200 images in total.


```
$ ./lib/organisedataset.sh
```


## Create 3 models based on 5 classes classification and 255 x 255 x 1 image input
```
python3 lib/createmodel.py pathmodelfolder modelname numberofclass imageheightsize imagewidthsize imagedepth
```
example
```
python3 /TechnicalAnalysisLearningwithResnet/lib/createmodel.py /TechnicalAnalysisLearningwithResnet/model/ init 5 255 255 1 
```

return 3 models based on resnet, vgg and a personalized one

example for resnet model

![example for resnet model](https://github.com/imiled/TechnicalAnalysisLearningwithResnet/blob/main/docs/resnetmodel.png)
## Train the model based on the dataset for training
```
python3 lib/train.py folderpathwithcategorizedfolderimages pathformodel nameofmodelinput nameoftrainedmodeloutput loss optimizer metric nbofclasses batchsize nbofepochs learningrate
```

example

```
python3 /TechnicalAnalysisLearningwithResnet/lib/train.py /TechnicalAnalysisLearningwithResnet/images/imagestrain/ \
                      /content/TechnicalAnalysisLearningwithResnet/model/ \
                      simplemodel_init.h5 \
                      simplemodel_trained \
                      'categorical_crossentropy' 'adam' 'accuracy' 5 32 20 0.01 \
```

This part will train the NN and save the trained model in the correct path. but t forst step it load the dataset for training split it in 2 for validation and arrange it to a 255 x 255 x 1 format with greyscale 

You can refer to the colab notebook so as to find the different setup tested to get to our accepable trained model through SGD on resnet.

![alt text](https://github.com/imiled/TechnicalAnalysisLearningwithResnet/blob/main/docs/accuracy.png)


## Test : generating the confusion matrix on dataset spared for testing
```
python3 lib/test.py pathforimagetotest fullpathofthemodel pathfortheresulttobesaved
```

example

```
python3 lib/test.py /content/TechnicalAnalysisLearningwithResnet/images/imagestest/ \
  /content/TechnicalAnalysisLearningwithResnet/model/TL_resnet_trained4.h5 \
  /content/TechnicalAnalysisLearningwithResnet/results/ 5
```  

This part will evaluate the model with the testing dataset that we generated in first step.
We show the accuracy, the confusion matrix and the classification report in the folder pathfortheresulttobesaved 

![alt text](https://github.com/imiled/TechnicalAnalysisLearningwithResnet/blob/main/docs/resulttest.png)


## Run : Guess future market state from random image
```
python3 lib/run.py pathforimagetopredit pathofthemodel pathfortheresults
```

example

```
python3 lib/run.py /content/TechnicalAnalysisLearningwithResnet/images/imagestrain/NN/ \
  /content/TechnicalAnalysisLearningwithResnet/model/TL_resnet_trained4.h5 \
  /content/TechnicalAnalysisLearningwithResnet/results/
```

Take an image of an historical graph from a market webpage like investing.com, crop the image to only fit the graph and save it to a folder for example with name image1.PNG or give the full path of the image when asked.
This execution tell us which market state in the future is the best representative in the file found in the resultpath with name

result_run_on_images%Y%m%d-%H%M%S.csv

![alt text](https://github.com/imiled/TechnicalAnalysisLearningwithResnet/blob/main/docs/runresult.png)
