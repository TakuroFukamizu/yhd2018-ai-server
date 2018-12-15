# YHD2018 AI server


## prep

```sh
$ pipenv install
```

## config

create `.env` file.  
```sh
MQ_HOST=192.168.179.7
MQ_PORT=1883
YOLO_MODEL_PATH=model_data/origin/yolo-tiny.h5
YOLO_ANCHORS_PATH=model_data/tiny_yolo_anchors.txt
YOLO_CLASSES_PATH=model_data/origin/labels.txt
```

## start server

```sh
$ pipenv run python3 src/app.py 
```
