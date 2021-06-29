## Setup
+ For all our experiments we build upon the Deep Learning AMI provided by AWS.
+ Start an instance with Deep Learning AMI.
+ Download imagenet data and follow the preprocessing instructions from (here)[https://github.com/pytorch/examples/tree/master/imagenet]
+ Next for BERT Download the Sogou News dataset follow the original repo to prepare dataset and BERT models available [here](https://github.com/xuyige/BERT4doc-Classification/blob/master/README.md).

Data available at: [here](https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM).

Some additional datasets available at: [here](https://course.fast.ai/datasets).

Models available at: 

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

+ Next create an AMI using the instructions provided [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html) 
+ Also create a new IAM role which will provide your EC2 instances permission to
  write to S3

## Running the code

The code borrows a lot of structure from Thijis Vogels's code for PowerSGD.
+ To launch the code look at launch_ec2_run_commands.py 
+ You also need to provide the bash file which will eventually launch the code.
+ Look at run_ddp.sh and run_dpp_var_bandwidth.sh for model which use imagenet
  dataset.
+ Look at run_ddp_bert.sh for running BERT model. 
+ launch_ec2_run_commands.py when provided the github repository with code and
  the bash file will automatically launch the code and write the data to S3

+ The code to run is provided in main_ddp_final.py and main_bert.py


