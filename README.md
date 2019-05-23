# Sequence-to-Sequence Video Object Segmentation
An end-to-end trainable model for object segmentation using convolutional LSTM and VGG-16 Architecture.

### Abstract:
This is an implementation of the method based on Conv-LSTM and Encoder-Decoder architecture proposed in:

https://arxiv.org/abs/1809.00461 

### Youtube-VOS dataset:
YouTube-VOS is the first large-scale benchmark that supports multiple video object segmentation tasks. with4000+ high-resolution YouTube videos. It can be downloaded from:

https://youtube-vos.org/

our results have been submitted to the leaderboard of the 2018 challenge:

https://competitions.codalab.org/competitions/19544#results

## Run Instructions:
### System Requirements:
The script is written in Python 3.5.2 and TensorFlow 1.13.1 for GPU, CPU operation might be possible but is not tested.
Required packages are:
```
tensorflow
yaml
json
time
math
json
cv2
shutil
random
os
numpy
cv2
PIL
shutil
```

### Configurations:
Please navigate to the `config.yaml` file in the root directory to setup the configurations:
```
$nano ./config.yaml
```
The file contents following configurable variables:
```
configs:
    path: ../dataset/ # Path to the Youtube-VOS dataset downloaded from https://youtube-vos.org/ website>
    checkpoints_path: "./checkpoints/model-1" # Path to the pre-saved checdkpoints if you would like to fine-tune or evaluate.
```

### Training phase:
To initiate the training phase please run the following script:
```
$cd ./ECCV_Youtube_VOS
$python3 VOS.py --n_epochs=<number of epochs of training> --batch_Size=<size of the mini batch> --lr=<learning rate>
```

The program will start generating a pre-processed version of the Youtube-VOS dataset for the purpose of training. This copy can be deleted from your drive after training is done.

### Evaluation phase:
After training your model you can creat object masks for the set of images in the validation set of Youtube-VOS.
```
$cd ./ECCV_Youtube_VOS
$python3 VOS_evaluate.py --batch_size=<size of the mini batch> --scenario-name=<the name of the directory to save created object masks>
```

## Notes:
The model typically converges after 80 epochs with learning rate of 1e-5, the loss function is typically similar to the picture below:
![Loss History](/utils/loss_history.png?raw=true "Optional Title")




