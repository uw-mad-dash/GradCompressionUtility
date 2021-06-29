import sys
import boto3
import time

from pssh.clients import ParallelSSHClient

launch_cfg = {
    "name": "large_scale_verification",
    "key_name": "",
    "key_path" : "",
    "method": "spot",
    "region": "us-east-1",
    "az": "us-east-1b",
    # "ami_id": ""
    "ami_id": "",
    "spot_price": "4.5",
    "ssh_username": "ubuntu",
    "iam_role": "",
    # "path_to_keyfile": ,
    "instance_type": "p3.8xlarge",
    "instance_count": 16,
}


def launch_instances():
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])

    instance_lifecycle = launch_cfg['method']
    instance_count = launch_cfg['instance_count']
    launch_dict = {"KeyName": launch_cfg['key_name'],
                   "ImageId": launch_cfg['ami_id'],
                   "InstanceType": launch_cfg['instance_type'],
                   "Placement": {"AvailabilityZone":launch_cfg['az']},
                   "SecurityGroups": ["pytorch-distributed"],
                   "IamInstanceProfile":{'Name': launch_cfg['iam_role']}
                   }

    if instance_lifecycle == "spot":
        response = client.request_spot_instances(
                InstanceCount=launch_cfg['instance_count'],
                LaunchSpecification=launch_dict,
                SpotPrice=launch_cfg['spot_price'],
        )
        print (response)
    else:
        print("Spot is not being used")
        sys.exit()

    request_ids = list()
    for request in response['SpotInstanceRequests']:
        request_ids.append(request['SpotInstanceRequestId'])
    
    
    fulfilled_instances = list()
    loop = True
    
    print("Waiting for requests to fulfill")
    time.sleep(5)
    while loop:
        request = client.describe_spot_instance_requests(
            SpotInstanceRequestIds=request_ids)
        for req in request['SpotInstanceRequests']:
            print (req)
            if req['State'] in ['closed', 'cancelled', 'failed']:
                print ("{}:{}".format(req['SpotInstanceRequestId'],
                                          req['State']))
                loop = False
                break
            if 'InstanceId' in req and req['InstanceId']:
                fulfilled_instances.append(req['InstanceId'])
                print (req['InstanceId']+ 'running...')
        if len(fulfilled_instances) == launch_cfg['instance_count']:
            print("All requested instances are fulfilled")
            break
        time.sleep(5)
    if loop == False:
        print ("Unable to fulfill all requested instance ..")
        sys.exit()

    while loop:
        loop = False
        response = client.describe_instance_status(
            InstanceIds=fulfilled_instances) 
        for status in response['InstanceStatuses']:
            if status['InstanceType']['Name'] != 'running':
                loop = True
    print ('All instances are running ..')
    
    
    #getting host keys

    instance_collection = ec2.instances.filter(Filters=[{'Name':'instance-id',
                                                         'Values':
                                                         fulfilled_instances}])
    private_ip = []
    public_ip = []
    for instance in instance_collection:
        print(instance.private_ip_address)
        private_ip.append(instance.private_ip_address)
        print (instance.public_ip_address)
        public_ip.append(instance.public_ip_address)
    return (private_ip, public_ip, fulfilled_instances)


def terminate_instances(instance_id):
    print ("Terminating instances ....")
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])
    instance_collection = ec2.instances.filter(Filters=[{'Name': 'instance-id',
                                                         'Values':
                                                         instance_id}])
    for instance in instance_collection:
        instance.terminate()
    print ("Bye Bye instances ...") 

def run_large_scale():
    private_ip, public_ip, instance_ids = launch_instances()
    time.sleep(1)
    # host_name = ["ubuntu@{}".format(i) for i in public_ip]
    client = ParallelSSHClient(public_ip, user="ubuntu", pkey=launch_cfg['key_path'])


    # bash run.sh resnet50 tcp://172.31.70.9:2345 0 64 /home/ubuntu/imagenet_data
    # cuda:0 temp PowerSGD 4 2  powersgd_rank_4_bsize_64_2machine
    
    run_args = [{'cmd': "git clone repo_code&& cd compression_imagenet_code && bash run_ddp_ps.sh {} {} 64 /home/ubuntu/imagenet_data trial {} ps_resnets_64 ".format(private_ip[0],i, len(private_ip))} for i in range(launch_cfg['instance_count'])]
    print (run_args) 
    output = client.run_command('%(cmd)s', host_args=run_args)
    
    for hosts_out in output:
        for line in hosts_out.stdout:
            print (line)

        for line in hosts_out.stderr:
            print (line)

    client.join(consume_output=True)
    terminate_instances(instance_ids)

if __name__ == "__main__":
    run_large_scale()

