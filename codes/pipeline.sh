# python train_lgbm.py feat.lag_win_pairs=[]
# python train_lgbm.py feat.lag_win_pairs=[[56,28],[28,28],[28,56],[7,7],[1,28]]
# python train_lgbm.py feat.lag_win_pairs=[[56,28],[28,28],[28,56],[7,7],[1,14],[1,28]]
# python train_lgbm.py feat.lag_win_pairs=[[56,28],[28,28],[28,56],[7,7],[1,7],[1,14],[1,28]]

# python retrain_lgbm.py feat.lag_win_pairs=[[56,28],[28,28],[28,56],[7,7],[1,7],[1,14],[1,28]]

# python train_lgbm.py feat.lag_win_pairs=[[28,28],[7,7],[3,3],[1,7],[1,28]]
# python train_lgbm.py feat.lag_win_pairs=[[28,28],[7,7],[1,4],[1,7],[1,28]]
# python train_lgbm.py feat.lag_win_pairs=[[28,28],[7,7],[1,5],[1,7],[1,28]]


python retrain_lgbm.py feat.lag_win_pairs=[[28,28],[7,7],[3,3],[1,4],[1,7],[1,14],[1,28]]
python predict_private.py 
