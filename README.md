# Keypoint Detection

## Preparation
Put your data in './dataset'. Each dataset should contain
1. `rawImage`: folder contains all the input images.
2. `OutLabelTxt`: folder contains all the annotations in `txt` format. 

## How to train from scratch
1. run `prepare_data.m` to generate json file
2. run `python prepare_LMDB.py` to generate LMDB
3. run `python prepare_proto.py` to generate protofiles
4. run `./prototxt/${dbname}/train.sh` to train the caffe model

