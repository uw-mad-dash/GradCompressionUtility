#!/bin/bash
arch=$1
master_ip=$2
rank=$3
bsize=$4
dataset_location=$5
device=$6
log_file=$7
reducer=$8
reducer_param=$9
num_workers=${10}
s3_prefix=${11}

echo $arch;
echo "$master_ip";
echo "$rank";
echo "$reducer";
echo "$bsize";
echo "$dataset_location";
echo "$device";
echo "$log_file";
echo "$reducer";
echo "$reducer_param";
echo "$num_workers";
echo $s3_prefix;
#./run.sh -arch resnet18 -master-ip tcp://127.0.0.1:2345 -rank 0 -reducer powersgd -bsize 28 -dataset-location /home/ubunut -device cuda:0 -log-file temp -reducer powersgd -reducer-param 2
source activate pytorch_p36
python main.py --arch $arch --master-ip $master_ip --rank $rank --reducer $reducer --batch-size $bsize --dataset-location $dataset_location --device $device --log-file $log_file --reducer-param $reducer_param --num-workers $num_workers --s3-prefix $s3_prefix


# python main.py --arch $arch --master-ip $2 --rank $3 --reducer $4 --batch-size $5 --dataset-location $6 --device cuda:1 --log-file $8 --reducer $9 --reducer-param $reducer_param
