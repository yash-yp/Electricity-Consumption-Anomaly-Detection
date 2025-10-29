# Electricity-Consumption-Anomaly-Detection

The repository contains ML models for detection of electricity fraud. 

## Follow the given steps to run this code:

1.) Import the project on [Google Colab](https://colab.research.google.com/)

```
!git clone https://github.com/yash-yp/Electricity-Consumption-Anomaly-Detection
!mv /content/Electricity-Consumption-Anomaly-Detection/* ./
```

2.) Unzip the dataset

```
!7za e data.7z
```

3.) Run [preCode.py](preCode.py) to obtain the datasets for visualizations, training and testing.

```
!python preCode.py
```

4.) Uncomment the required the model from [models.py](models.py) and run it to get the required results

```
!python models.py
```
