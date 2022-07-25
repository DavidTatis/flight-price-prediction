# flight-price-prediction

Deep Neural Networks to predict fligh prices. I will apply this problem to my Hyperparameter tunning research using Meta learnign and Genetic Algorithms.
I create a dataset of DNN architectures and different hyperparameters and their respective performance -Mean Absolute Error (MAE) for prediction problems-.
I trained the models in parallel using GPUs. So, i create a script for each model but using the same Random GS configuration and Hyperparameters space.
<br/>
**Data set**:<br/>
https://www.kaggle.com/code/julienjta/flight-price-prediction-98-47-r2-score/data <br/>
<br/>
**Preprocessing**:<br/>
https://www.kaggle.com/code/julienjta/flight-price-prediction-98-47-r2-score <br/>


## Code:
I create the Random GS with help of the SciKit-learn library. The models using tensorflow.keras GPU version. <br/>
**FCUNet** Is a UNet architecture but using Fully Connected layers instead of convolutional layers.<br/>
**FCMnR** Merge and Run architecture using Fully Connected layers.<br/>
**IRNET** Inmediate residual layers using Fully Connected layers.<br/>
<br/>
For development and debugg purpose, i start coding in a Jupyter Notebook called fligh-prediction.ipynb. After a stable version, i create a python3.7 Script for each experiment/architecture. <br/>
<br/>
The final version of the codes are:<br/>
**  fcunet-flight-prediction.py**: FCUNet for the flight price prediction problem using a Random Grid search. <br/>
**  flight-prediction-script.py**: FCMnR for the flight price prediction problem using a Random Grid search.<br/>
**  IRNET-fcunet-flight-prediction.py**: IRNET for the flight price prediction problem using a Random Grid search.<br/>
  
 
