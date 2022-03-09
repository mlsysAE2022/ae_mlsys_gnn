source ~/.bashrc
conda init
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --remove channels defaults
conda info
conda config --set show_channel_urls yes

conda create -n pytorch python=3.8
source activate pytorch
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch torchvision torchaudio cudatoolkit=11.3 tqdm
conda install -c dglteam dgl
conda install -c conda-forge ninja
pip install auditwheel tqdm
pip install requests
conda clean --tarballs
