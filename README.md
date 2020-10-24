# TFM Introduction Deep Learning Tools For Finance 
## Application : Automated Financial Technical analysis using Resnet Neural Network

I focused in this project to apply a Transfert learning methodology of CNN to an image of the sp500 technical graph as an image. 

The objective is first to get a complete dataset, then a trained model based on Resnet infrastucture. 

In addition we analyse the model efficiency and last point for any image of a stock price historical graph we have a tool that tell us if we would rather buy, sell or hold.

You can find also in the docs/ repertory the TFM report and presentation with the google colab notebook. 

I consider here a problem of behaviour finance as most investor look throughly at those graph more than fondamental numbers and those graph can be interpretated on small horizon (minutes) or long (years) to get an estimation of its evolution. The humain will process this information deeply and the consequence of this process is the behaviour of stock market. Benjamin Graham in the "Intelligent Investor" - written in 1949 and considered as the bible of value investing - introduce the allegory of Mr. Market, meant to personify the irrationality and group-think of the stock market. As of august 2020, the value of some stocks are higher than the economy of France and Germany and some companies (Tesla) are bought at a price quite difficult to apprehend in terms of valuation fondamental and comparison to established company in Europe (Volkswagen) or Japan (Toyota). That is true, that our brain is set to always find an explanation but in this approach we' ll try to apprehend the impact of price evolution to make Mr Market more greedy or fearful.

In this project I have chosen to present 4 steps which can be taken separately as we can load save datas or models. 

I worked on it in google colab you can see it in the following file :

TechAnalysisLearningwithResnet.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imiled/MarketPredictionusingResnetandStockTechGraph/blob/main/docs/TechAnalysisLearningwithResnet.ipynb)

I recommend at start to use a virtual python envirenment using this specific command :
```
$ bash init.sh
```

and using this Docker command to get the appropiate environment base:
```
docker pull imiled/sp500from_image_to_state_prediction:1.0
```

But it can be launched in local also using the command to get the specific packages either for a cpu or a gpu :

for cpu :
```
pip install -r cpu_requirement.txt
```

for a gpu :
```
pip install -r gpu_requirement.txt
```


Now for each step can be taken independently as we are saving loading datas and model at each time.

## Generate Dataset of the Image and the Future maket state
```
python3 step1_generate_dataset_IndexImage.py
```

In this part we are generating the training and testing dataset.
First we download the historical prices of the sp500 from 1927 to 31 July 2020 and built the image of 15 days historical graph also we get the 5 days future price evolution of the sp500. 
From the future price evolution, we calculate a future state which can be splitted in 6 classes :

**Sell-Sell | Sell- Neutral | Neutral | Neutral -Buy | Buy -Buy |  Error**

The objective is to get the following files which represent a dataframe in the data/ repertory:

*X_train_image.csv , X_test_image.csv* a 3072 column time serie dataframe  of the image (32 x 32 x3) of the sp500 closing price 

*Y_test_StateClass.csv, Y_train_StateClass.csv* a 1 column time serie dataframe of the future state value betwwen -1 to 4

We generate also the following files but we won´t use it in this project - more for RNN & price prediction - *Y_test_FutPredict.csv Y_train_FutPredict.csv*

the testing and training time serie dataset are shuffled by the date of reference with a split number of 0.8

Please note that: 
1. We can increase the dataset taking into account the evolution very liquid stocks or other indices as long as we have very high the liquidity and number of participants 
2. The calculation of the dataset can take more than 6 hours of calulation as the code is not optimized so far, we can quickly implement parallel computing and rapid image setup instead of using matplotlib library

## Loading training datas and Build up of the VGGsp500 model and train
```
python3 step2_loadingtrainingdatas_vgg_transfert_modelandtraining.py
```

This part is for loading the training dataset as it is better to generate it once for all in step 1 because of it time consuming process.
This part also configure back the X_train datas from dataframe based on columns to a (32,32,3) np. array for the input of the model 

Then we apply the Transfert model methodology with vgg16 and some other layers.
We use for this example a categorical_crossentropy loss and rmsprop optimizer.
This part can be fined tuned for each financial index or stock index (layers, optimizer, metrics, dropout) but in this case we introduced a simplier case.
We train and save the model, please refer to XX to see the convergence of the model.

We have 14.7M parameters and 66k trainable parametres. the size of training input is 571M only for the image not including rolling volatility, moving average etc
In the Colab notebook you can see the tensorboard document so as to monitor the convergence of the training. 

## Evaluate the VGGsp500
```
python3 step3_evaluate_vggsp500_model.py
```
This part will evaluate the model with the testing dataset that we generated in first step.
We show the accuracy, the confusion matrix and the classification report 

## Guess future market state from random image
```
python3 step4_guess_future_marketstate_from_image.py
```

Take an image of an historical graph from a market webpage like investing.com, crop the image to only fit the graph and save it to the ImageM/ folder for example with name image1.PNG or give the full path of the image when asked.

This execution tell us which market state in the future is the best representative.

