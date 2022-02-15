rm figure10.csv
cd train
#gat no fusion
python train_gatconv_no_fusion.py --dataset=reddit --num-heads=4 --num-hidden=64 --epochs=100
#gat fusion but stash
python train_gatconv_stash.py --dataset=reddit --num-heads=4 --num-hidden=64 --epochs=100
#gat fusion and re-compute
python train_gatconv_our.py --dataset=reddit --num-heads=4 --num-hidden=64 --epochs=100 --output=figure10.csv
#monet no fusion
python train_gmmconv_dgl.py --dataset=reddit --n-epochs=100 --n-hidden=16 --n-kernels=2 --pseudo-dim=1 --output=figure10.csv --key=nofusion
#monet fusion but stash
python train_gmmconv_stash.py --dataset=reddit --n-epochs=100 --n-hidden=16 --n-kernels=2 --pseudo-dim=1
#monet fusion and re-compute
python train_gmmconv_our.py --dataset=reddit --n-epochs=100 --n-hidden=16 --n-kernels=2 --pseudo-dim=1 --output=figure10.csv
