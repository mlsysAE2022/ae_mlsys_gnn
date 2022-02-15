import os
import sys
import csv

file = 'config.txt'
with open(file, 'r') as f:
    exe = f.readline().strip('\n')
    cuda_home = f.readline()

fastgat_path = os.path.abspath(os.path.dirname(os.getcwd()))
result_folder = fastgat_path + '/result/'
os.path.exists(result_folder)
ncu = cuda_home + '/bin/nv-nsight-cu-cli'
metrics = ['dram__bytes_write.sum', 'dram__bytes_read.sum']
folder = ['write', 'read']
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def profile_edgeconv_train():
    k = ['20', '40']
    bs = ['32', '64']
    os.chdir("./train")
    train_ = "train/"
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            nl = name.split('_')
            for bbss in bs:
                for kk in k:
                    for met in metrics:
                        parser = ' --batch-size ' + bbss + ' --k ' + kk
                        if nl[1] == 'edgeconv':
                            if nl[2] == 'dgl.py' or nl[2] == 'our.py':
                                os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}train/{folder[metrics.index(met)]}/{nl[1]}_{kk}{bbss}_{nl[2]}.csv {exe} {name} --profileio 1 {parser}')
    os.chdir("..")

def profile_gmmconv_train():
    dataset = ['cora', 'pubmed', 'citeseer', 'reddit']
    # train
    train_ = "train/"
    os.chdir("./train")
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            nl = name.split('_')
            for da in dataset:
                for met in metrics:
                    parser = ' '
                    if nl[1] == 'gmmconv':
                        if nl[2] == 'dgl.py' or nl[2] == 'our.py':
                            parser += '--dataset ' + da + ' --gpu 0'
                            if da == 'pubmed':
                                parser += ' --pseudo-dim 3'
                            if da == 'reddit':
                                parser += ' --n-kernels 2 --pseudo-dim 1'
                            os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}train/{folder[metrics.index(met)]}/{nl[1]}_{da}_{nl[2]}.csv {exe} {name} --profileio 1 {parser}')
    os.chdir("..")

def profile_gatconv_train():
    dataset = ['cora', 'pubmed', 'citeseer', 'reddit']
    os.chdir("./train")
    train_ = "train/"
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            nl = name.split('_')
            for da in dataset:
                for met in metrics:
                    parser = ' '
                    if nl[1] == 'gatconv':
                        if nl[2] == 'dgl.py' or nl[2] == 'our.py':
                            parser += '--dataset ' + da + ' --gpu 0' + ' --num-heads 1' + ' --num-hidden 128'
                            os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}train/{folder[metrics.index(met)]}/{nl[1]}_{da}_{nl[2]}.csv {exe} {name} --profileio 1 {parser}' )
    os.chdir("..")

def filter_train_io():
    start = 0
    gatdict = dict()
    gmmdict = dict()
    edgedict = dict()
    output_file = 'figure7_io.csv'
    # read
    os.chdir("../result/train/read")
    for root, dirs, files in os.walk(".", topdown=False):
        os.system("pwd")
        for name in files:
            accio = 0
            with open(name, newline = '')  as f:
                reader = csv.reader(f)
                for row in reader:
                    if(len(row) > 2):
                        if(row[4] == "start()"):
                            start = 1
                        if(row[4] == "end()"):
                            start = 0
                        if(start == 1):
                            x = filter(str.isdigit, row[-1])
                            x = int("".join(list(x)))
                            accio += int(x)
            if name is not None:
                nl = name.split('_')
                method = nl[2].split('.')[0]
                if nl[0] == 'gatconv':
                    gatdict[nl[1]+'_'+method] = accio/(1024**3)
                elif nl[0] == 'gmmconv':
                    gmmdict[nl[1]+'_'+method] = accio/(1024**3)
                elif nl[0] == 'edgeconv':
                    edgedict[nl[1]+'_'+method] = accio/(1024**3)
    # write
    start = 0
    os.chdir("../write")
    for root, dirs, files in os.walk(".", topdown=False):
        os.system("pwd")
        for name in files:
            accio = 0
            with open(name, newline = '')  as f:
                reader = csv.reader(f)
                for row in reader:
                    if(len(row) > 2):
                        if(row[4] == "start()"):
                            start = 1
                        if(row[4] == "end()"):
                            start = 0
                        if(start == 1):
                            x = filter(str.isdigit, row[-1])
                            x = int("".join(list(x)))
                            accio += int(x)
            if name is not None:
                nl = name.split('_')
                method = nl[2].split('.')[0]
                
                if nl[0] == 'gatconv':
                    gatdict[nl[1]+'_'+method] += accio/(1024**3)
                elif nl[0] == 'gmmconv':
                    gmmdict[nl[1]+'_'+method] += accio/(1024**3)
                elif nl[0] == 'edgeconv':
                    edgedict[nl[1]+'_'+method] += accio/(1024**3)

    os.chdir("../../../script")
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter = ',')
        for i, k in gatdict.items():
            sp = i.split('_')
            writer.writerow(['gatconv', sp[1], sp[0], k])

        for i, k in gmmdict.items():
            sp = i.split('_')
            writer.writerow(['gmmconv', sp[1], sp[0], k])

        for i, k in edgedict.items():
            sp = i.split('_')
            writer.writerow(['edgeconv', sp[1], sp[0], k])


if __name__ == '__main__':
    profile_gatconv_train()
    profile_gmmconv_train()
    profile_edgeconv_train()
    filter_train_io()