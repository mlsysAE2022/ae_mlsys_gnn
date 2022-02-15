rm figure11_3090.csv
cd train
python train_gatconv_our.py --dataset=reddit --num-heads=4 --num-hidden=64 --epochs=100 --output=figure11_3090.csv
python train_gatconv_dgl.py --dataset=reddit --num-heads=4 --num-hidden=64 --epochs=100 --output=figure11_3090.csv
python train_edgeconv_our.py --k=40 --batch-size=64 --output=figure11_3090.csv
python train_edgeconv_dgl.py --k=40 --batch-size=64 --output=figure11_3090.csv
python train_gmmconv_our.py --dataset=reddit --n-epochs=100 --n-hidden=16 --n-kernels=2 --pseudo-dim=1 --output=figure11_3090.csv
python train_gmmconv_dgl.py --dataset=reddit --n-epochs=100 --n-hidden=16 --n-kernels=2 --pseudo-dim=1 --output=figure11_3090.csv