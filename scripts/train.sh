
set -ex

python3 src/preprocess.py process-file --infile data/atec/train.csv --outfile data/atec/train_seg.json --exchange True
python3 src/preprocess.py process-file --infile data/atec/dev.csv --outfile data/atec/dev_seg.json
python3 src/preprocess.py process-file  --infile data/atec/test.csv --outfile data/atec/test_seg.json
python3 src/wordemb.py

python3 src/train.py --model-prefix std_model
