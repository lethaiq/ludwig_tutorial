# Ludiwg Tutorial
Please always first refer to ```User Guide``` of Ludwig from ```https://uber.github.io/ludwig/user_guide/``` when you have any questions.

## Preparation
  
- Create a new folder called: ```ludwig_exercise``` in your Google Drive.  
- Download zip file: ```clickbait.csv``` and ```config.yaml``` in to the ```ludwig_exercise``` folder created in the previous step.  
- Make sure you have at least 1GB free space in your Google Drive

## Clickbait Detection

### Create new Google Collab
- Go to your Google Drive
- Access Google Collab at: https://colab.research.google.com. Click "New Python3 Notebook", the browser will show the interface of a iPython Notebook
- Add GPU: Click menu "Edit" -> "Notebook Settings". In the dialog box, select "GPU" from drop box "Hardware Acceleration"

### Making sure GPU is loaded:

```
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

### Install ludwig: 

```!apt-get install libgmp-dev libmpfr-dev libmpc-dev```  
```!pip install ludwig --no-deps tensorflow```

### Import libraries:

```import pandas as pd```  
```import numpy as np```  
```from google.colab import drive```  
	
### Mount Google Drive folder to the python environment:
```drive.mount('/gdrive')```  

Click the authorization link provided, follow the instruction to get the authorization code. Copy the authorization code back to the Google Collab notebook and press enter. Now it should show: **Mounted at /gdrive**

	
### Access the ludwig_exercise folder:
```%cd /gdrive/My Drive/ludwig_exercise```
```%ls```  

Making sure you can see files: ```clickbait.csv```,  ```config.yaml``` show up

### Load clickbait dataset:
```df = pd.read_csv('clickbait.csv', index_col=0)```  

### Examine the dataset:
```df.head()```

### Print out some statistics:
```print("total data samples:{}".format(df.shape[0]))```  
```print("label distribution:{}".format(np.bincount(df['class'])))```
	
### Run training using ludwig:
Now we want to command ludwig to do training experiment on the clickbait dataset ```clickbait.csv``` with a deep learning architecture described in ```config.yaml```  

```
!ludwig experiment \
  --data_csv 'clickbait.csv' \
  --model_definition_file 'config.yaml'
```

Ludwig will automatically split the whole dataset into 3 parts: train, validation and test sets, in ratio: 75%:15%:10%. You can manually split the dataset by 1) create a new column called ```split``` in the dataset, 2) assign the split column to value ```0``` for training samples, ```1``` for validataion samples and ```2``` for test samples.


## Debugs

If at any point in the the Google Collab shows error, you can reset the running time: menu "Runtime" -> "Reset all runtimes…", then run every cells again

## Exercise

### Sentence Similarity
```quora dataset```: I have also prepared two datasets: ```quora_train.csv``` and ```quora_test.csv``` from this link: ```https://drive.google.com/drive/folders/1VyX0Tf5UZ3SzWEE8wYiMztRbZWavttaq?usp=sharing```. Details of this task can be found at: ```https://www.kaggle.com/c/quora-question-pairs```. Please use the train dataset to train a model to predict question pairs similiarity, and report on test set.

Hint 1: ```Siamese architecture``` at ```https://uber.github.io/ludwig/examples/#one-shot-learning-with-siamese-networks``` is one example of neural network model that can be used. In this example they want to predict similarity between a pair of images, we need to change the architecture accordingly to accommodate our task.

Hint 2: try different encoders

Hint 3: try pre-trained word-embedding such as glove or word2vec with ```pretrained_embeddings``` parameter on the configuration file (prefer to user manual for details)

Hint 4: try different training parameters (batch-size, learning rate, etc.)

