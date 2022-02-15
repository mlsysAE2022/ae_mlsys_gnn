rm figure11_2080.csv
cd train
python train_gatconv_our.py --dataset=reddit --num-heads=4 --num-hidden=64 --epochs=100 --output=figure11_2080.csv
python train_edgeconv_our.py --k=40 --batch-size=64 --output=figure11_2080.csv
python train_gmmconv_our.py --dataset=reddit --n-epochs=100 --n-hidden=16 --n-kernels=2 --pseudo-dim=1 --output=figure11_2080.csv