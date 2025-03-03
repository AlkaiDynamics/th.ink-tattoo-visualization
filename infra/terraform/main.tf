provider "aws" {
  region = "us-west-2"  # Specify your desired AWS region
}

resource "aws_vpc" "think_vpc" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "think_vpc"
  }
}

resource "aws_subnet" "think_subnet" {
  vpc_id     = aws_vpc.think_vpc.id
  cidr_block = "10.0.1.0/24"
  availability_zone = "us-west-2a"  # Adjust as needed

  tags = {
    Name = "think_subnet"
  }
}

resource "aws_internet_gateway" "think_internet_gateway" {
  vpc_id = aws_vpc.think_vpc.id

  tags = {
    Name = "think_internet_gateway"
  }
}

resource "aws_route_table" "think_route_table" {
  vpc_id = aws_vpc.think_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.think_internet_gateway.id
  }

  tags = {
    Name = "think_route_table"
  }
}

resource "aws_route_table_association" "think_route_table_association" {
  subnet_id      = aws_subnet.think_subnet.id
  route_table_id = aws_route_table.think_route_table.id
}

resource "aws_security_group" "think_sg" {
  name        = "think_sg"
  description = "Allow SSH and HTTP/HTTPS traffic"
  vpc_id      = aws_vpc.think_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Restrict in production
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "think_security_group"
  }
}

resource "aws_instance" "think_ec2" {
  ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2 AMI (HVM), SSD Volume Type
  instance_type = "t2.micro"  # Change based on requirements
  subnet_id     = aws_subnet.think_subnet.id
  security_groups = [aws_security_group.think_sg.name]

  associate_public_ip_address = true

  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y
              sudo amazon-linux-extras install docker -y
              sudo service docker start
              sudo usermod -a -G docker ec2-user
              sudo chkconfig docker on
              sudo docker run -d -p 8000:8000 your-dockerhub-username/think-backend:latest
              EOF

  tags = {
    Name = "think_ec2_instance"
  }
}

output "ec2_public_ip" {
  description = "The public IP of the Th.ink EC2 instance"
  value       = aws_instance.think_ec2.public_ip
}