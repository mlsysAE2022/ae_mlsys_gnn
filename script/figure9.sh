rm figure9.csv
cd inference
#gat without fusion
python gatconv_nofusion_forward.py --dataset=reddit --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100
#gat with fusion
python gatconv_our_forward.py --dataset=reddit --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100
#edgeconv without fusion
python edgeconv_nofusion_forward.py
#edgeconv with fusion
python edgeconv_our_forward.py
#monet without fusion
python gmmconv_dgl_forward.py --in-feat=16 --out-feat=16 --n-kernels=2 --pseudo-dim=1 --dataset=reddit --gpu=0
#monet with fusion
python gmmconv_our_forward.py --in-feat=16 --out-feat=16 --n-kernels=2 --pseudo-dim=1 --dataset=reddit --gpu=0