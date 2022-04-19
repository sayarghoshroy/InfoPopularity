# BERT Reg

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge

## Data Preparation For CNN/Dailymail and InfoPop

#### Step 1 Download InfoPop and CNN/DM data
Move the data files to the appropriate location. The files will be downloaded in the same link as the code. 

####  Step 2. Document Splitting

```
python3 split.py
```

* This is to split the documents so that the model can read all sentences. Change the input and output locations within split.py


####  Step 3. Format to Simpler Json Files
 
```
python3 convert.py
```
* Again change input and output location within convert.py. To convert R1, R2, RL data from CNN/DM, use j[1],j[2],j[3] respectively.

####  Step 4. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`). Save final file as f.bert.pt.

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

```
python train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0,1,2,3 -report_every 50 -save_checkpoint_steps 5000 -batch_size 3000 -train_steps 50000 -accum_count 2 -use_interval true -warmup_steps 10000 -max_pos 1536
```

## Model Validation
```
 python train.py -task ext -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -test_from MODEL_PATH -sep_optim true -use_interval true -visible_gpus 0,1,2,3 -max_pos 1536 -alpha 0.95 -result_path RES 
```

## Model Evaluation
```
 python train.py -task ext -mode validate -test_all -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -test_from MODEL_PATH -sep_optim true -use_interval true -visible_gpus 0,1,2,3 -max_pos 1536 -alpha 0.95 -result_path RES 
```
