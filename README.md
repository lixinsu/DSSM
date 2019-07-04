# basic text matching model
## representation-based method

基本的文本匹配模型，模型使用词语和字符两种级别的级别的嵌入向量，将；两段文本进行表示，然后拼接得到隐含向量，进行二分类。

features
 - pretrained word and char embedding
 - combine word-level and char-level matching signal

 ## training
   - clone the reposity and run `sh scripts/setup.sh` 
   - `cd data/atec` and download data from - [atec data](https://dc.cloud.alipay.com/index#/home)
   - split the origin csv file to `train.csv` , `dev.csv` and `test.csv`
   - run `sh scripts/train.sh`
   
   
