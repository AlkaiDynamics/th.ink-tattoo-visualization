import os
import sys
import time
import requests
import boto3
from datetime import datetime

# Configuration
API_URL = os.getenv('API_URL', 'https://your-api-url.com/metrics')
QPS_THRESHOLD = 10000
INSTANCE_TYPE = 't2.micro'  # Change based on requirements
AUTO_SCALING_GROUP = os.getenv('AUTO_SCALING_GROUP')
REGION = 'us-west-2'

# Initialize AWS clients
autoscaling_client = boto3.client('autoscaling', region_name=REGION)
cloudwatch_client = boto3.client('cloudwatch', region_name=REGION)

def get_qps():
    try:
        response = requests.get(f"{API_URL}/qps")
        if response.status_code == 200:
            return int(response.json().get('qps', 0))
        else:
            print(f"Failed to get QPS: {response.status_code}")
            return 0
    except Exception as e:
        print(f"Error fetching QPS: {e}")
        return 0

def scale_up():
    try:
        response = autoscaling_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[AUTO_SCALING_GROUP]
        )
        asg = response['AutoScalingGroups'][0]
        current_size = asg['DesiredCapacity']
        new_size = current_size + 1
        autoscaling_client.set_desired_capacity(
            AutoScalingGroupName=AUTO_SCALING_GROUP,
            DesiredCapacity=new_size,
            HonorCooldown=True
        )
        print(f"Scaled up Auto Scaling Group to {new_size}")
    except Exception as e:
        print(f"Error scaling up: {e}")

def scale_down():
    try:
        response = autoscaling_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[AUTO_SCALING_GROUP]
        )
        asg = response['AutoScalingGroups'][0]
        current_size = asg['DesiredCapacity']
        if current_size > 1:
            new_size = current_size - 1
            autoscaling_client.set_desired_capacity(
                AutoScalingGroupName=AUTO_SCALING_GROUP,
                DesiredCapacity=new_size,
                HonorCooldown=True
            )
            print(f"Scaled down Auto Scaling Group to {new_size}")
        else:
            print("Auto Scaling Group is already at minimum size.")
    except Exception as e:
        print(f"Error scaling down: {e}")

def monitor():
    while True:
        qps = get_qps()
        print(f"[{datetime.utcnow()}] Current QPS: {qps}")
        
        if qps > QPS_THRESHOLD:
            scale_up()
        elif qps < QPS_THRESHOLD / 2:
            scale_down()
        
        # Sleep for a defined interval before next check
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    if not AUTO_SCALING_GROUP:
        print("AUTO_SCALING_GROUP environment variable not set.")
        sys.exit(1)
    
    monitor()