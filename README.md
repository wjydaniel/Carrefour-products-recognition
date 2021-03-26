# Carrefour_products_recognition

This project is a try to do carrefour products photos recognition using pre-trained ResNeXt50 Model on google colab.

### Download data from google drive to colab
```
from google.colab import drive
drive.mount('./gdrive')
```

### Create tensorboard
```
%reload_ext tensorboard
```

### Look at the model
```
!python model.py
```

### Training
```
!python './gdrive/MyDrive/data/train.py'(to be adapted) --data_dir './gdrive/MyDrive/data'(to be adapted) --class_name 'all' --tensorboard tf_log
```

### Testing
```
!python './gdrive/MyDrive/data/test.py'(to be adapted) --data_dir './gdrive/MyDrive/data'(to be adapted) --resume './gdrive/MyDrive/data/epoch000_0.01582_0.4599.pth'(to be adapted)
```

### Show training performance
```
tensorboard --logdir=tf_log
```
