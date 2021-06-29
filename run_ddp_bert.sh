#!/bin/bash
master_ip=$1
rank=$2
bsize=$3
dataset_location=$4
log_file=$5
num_workers=${6}
s3_prefix=${7}

echo "$master_ip";
echo "$rank";
echo "$bsize";
echo "$dataset_location";
echo "$log_file";
echo "$num_workers";
echo $s3_prefix;
#./run.sh -arch resnet18 -master-ip tcp://127.0.0.1:2345 -rank 0 -reducer powersgd -bsize 28 -dataset-location /home/ubunut -device cuda:0 -log-file temp -reducer powersgd -reducer-param 2
source activate pytorch_latest_p37
OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$num_workers --node_rank=$rank --master_addr=$master_ip --master_port=2345 main_bert.py --batch-size $bsize --dataset-location $dataset_location --log-file $log_file --s3-prefix $s3_prefix --node_rank $rank --max_seq_length 512


# python main.py --arch $arch --master-ip $2 --rank $3 --reducer $4 --batch-size $5 --dataset-location $6 --device cuda:1 --log-file $8 --reducer $9 --reducer-param $reducer_param
