import os
import sys
import csv

file = 'config.txt'
with open(file, 'r') as f:
    exe = f.readline().strip('\n')
    cuda_home = f.readline()

metrics = ['dram__bytes_write.sum', 'dram__bytes_read.sum']
folder = ['write', 'read']
fastgat_path = os.path.abspath(os.path.dirname(os.getcwd()))
result_folder = fastgat_path + '/result/'
os.path.exists(result_folder)
ncu = cuda_home + '/bin/nv-nsight-cu-cli'
proio = "--profileio 1"

def profile_figure8():
    os.chdir("./inference")
    for met in metrics:
        os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}figure8/{folder[metrics.index(met)]}/gatconv_withoutreorg.csv {exe} gatconv_000_forward.py --dataset=pubmed --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100 {proio}')
        os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}figure8/{folder[metrics.index(met)]}/gatconv_withreorg.csv {exe} gatconv_dgl_forward.py --dataset=pubmed --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100 {proio}')
        os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}figure8/{folder[metrics.index(met)]}/edgeconv_withoutreorg.csv {exe} edgeconv_dgl_forward.py {proio}')
        os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}figure8/{folder[metrics.index(met)]}/edgeconv_withreorg.csv {exe} edgeconv_reorg_forward.py {proio}')
    os.chdir("..")

def profile_figure9():
    os.chdir("./inference")
    for met in metrics:
        os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}figure9/{folder[metrics.index(met)]}/gatconv_nofusion.csv {exe} gatconv_nofusion_forward.py --dataset=reddit --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100 {proio}')
        os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}figure9/{folder[metrics.index(met)]}/gatconv_our.csv {exe} gatconv_our_forward.py --dataset=reddit --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100 {proio}')
        os.system(f'sudo {ncu} --metrics {met} --csv --log-file {result_folder}figure9/{folder[metrics.index(met)]}/edgeconv_nofusion.csv {exe} edgeconv_nofusion_forward.py {proio}')
    os.chdir("..")

def filter_figure_io(figurefile):
    start = 0
    conv_dict = dict()
    # read
    dirres = "../result/" + figurefile + "/read"
    os.chdir(dirres)
    for root, dirs, files in os.walk(".", topdown=False):
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
                nl = name.split('.')[0]
                conv_dict[nl] = accio/(1024**3)

    os.chdir("../write")
    for root, dirs, files in os.walk(".", topdown=False):
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
                nl = name.split('.')[0]
                conv_dict[nl] += accio/(1024**3)

    os.chdir("../../../script")
    with open(figurefile + '_io.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter = ',')
        for i, k in conv_dict.items():
            sp = i.split('_')
            writer.writerow([sp[0], sp[1], k])

if __name__ == "__main__":
    profile_figure8()
    profile_figure9()
    filter_figure_io("figure8")
    filter_figure_io("figure9")