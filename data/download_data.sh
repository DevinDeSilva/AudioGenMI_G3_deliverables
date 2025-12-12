#!/bin/bash
curl -L -o vctk-corpus.zip https://www.kaggle.com/api/v1/datasets/download/pratt3000/vctk-corpus
unzip vctk-corpus.zip

curl -L -o darpa-timit-acousticphonetic-continuous-speech.zip https://www.kaggle.com/api/v1/datasets/download/mfekadu/darpa-timit-acousticphonetic-continuous-speech
unzip darpa-timit-acousticphonetic-continuous-speech.zip 

wget https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip
unzip vox1_dev_wav.zip