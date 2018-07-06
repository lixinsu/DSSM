# DSSM
DSSM for atec competition about question identity, reference 冲刺到NLP赛题第一名bird开源起步框架

performance 

- local test F1 0.5

features

 - pretrained word char embedding
 - word char double channel


 ## training
   - clone the reposity and run `sh scripts/setup.sh` 
   - `cd data/atec` and download data from - [atec data](https://dc.cloud.alipay.com/index#/home)
   - split the origin csv file to `train.csv` , `dev.csv` and `test.csv`
   - run `sh scripts/train.sh`
   
   
