apt-get update
apt-get install -y wget bzip2 expect git build-essential gcc && rm -rf /var/lib/apt/lists/*
cd /home
wget --no-check-certificate -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Linux-x86_64.sh
chmod 777 Anaconda3-2021.05-Linux-x86_64.sh

