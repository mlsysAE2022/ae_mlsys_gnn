rm figure8.csv
cd inference
#gat without reorg
python gatconv_000_forward.py --dataset=pubmed --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100
#gat with reorg
python gatconv_dgl_forward.py --dataset=pubmed --in_feats=64 --out_feats=64 --num-heads=4 --epochs=100
#edgeconv without reorg
python edgeconv_dgl_forward.py
#edgeconv with reorg
python edgeconv_reorg_forward.py