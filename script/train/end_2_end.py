import os

os.system("rm ../figure7.csv")

#gat
head=1 
f=128
for dataset in ['cora','pubmed','citeseer','reddit']:
    for implementation in ['our','dgl']:   
        os.system("python train_gatconv_{}.py --dataset={} --num-heads={} --num-hidden={} --epochs=100".format(implementation,dataset,head,f))

#edgeconv
for k in [20,40]:
    for batch in [32,64]:
        for implementation in ['our','dgl']:
            os.system("python train_edgeconv_{}.py --k={} --batch-size={}".format(implementation,k,batch))

#monet
for implementation in ['our','dgl']:
    os.system("python train_gmmconv_{}.py --dataset=cora --n-epochs=100 --n-hidden=16 --n-kernels=3 --pseudo-dim=2".format(implementation))
    os.system("python train_gmmconv_{}.py --dataset=pubmed --n-epochs=100 --n-hidden=16 --n-kernels=3 --pseudo-dim=3".format(implementation))
    os.system("python train_gmmconv_{}.py --dataset=citeseer --n-epochs=100 --n-hidden=16 --n-kernels=3 --pseudo-dim=3".format(implementation))
    os.system("python train_gmmconv_{}.py --dataset=reddit --n-epochs=100 --n-hidden=16 --n-kernels=2 --pseudo-dim=1".format(implementation))

