# python NNtrain_GS.py --seed 77 --lr 0.0001 --numgs 100 --cuda 'cuda:0' --trainname 'bb7c' --rcsdir '/home/ljm/workspace/datasets/traintest' --valdir '/home/ljm/workspace/datasets/traintest'
# python NNtrain_GS.py --seed 77 --lr 0.0005 --numgs 100 --cuda 'cuda:0' --trainname 'bb7c' --rcsdir '/home/ljm/workspace/datasets/traintest' --valdir '/home/ljm/workspace/datasets/traintest'
# python NNtrain_GS.py --seed 77 --lr 0.0005 --numgs 50 --cuda 'cuda:0' --trainname 'bb7c' --rcsdir '/home/ljm/workspace/datasets/traintest' --valdir '/home/ljm/workspace/datasets/traintest'
# python NNtrain_GS.py --seed 77 --lr 0.0001 --numgs 50 --cuda 'cuda:0' --trainname 'bb7c' --rcsdir '/home/ljm/workspace/datasets/traintest' --valdir '/home/ljm/workspace/datasets/traintest'
# python NNtrain_GS.py --seed 77 --lr 0.0005 --numgs 20 --cuda 'cuda:0' --trainname 'bb7c' --rcsdir '/home/ljm/workspace/datasets/traintest' --valdir '/home/ljm/workspace/datasets/traintest'
# python NNtrain_GS.py --seed 77 --lr 0.0001 --numgs 20 --cuda 'cuda:0' --trainname 'bb7c' --rcsdir '/home/ljm/workspace/datasets/traintest' --valdir '/home/ljm/workspace/datasets/traintest'

# python NNtrain_arg.py --seed 7 --cuda 'cuda:0' --gama 0.0005 --trainname 'bb7c_abNone' --rcsdir '/home/ljm/workspace/datasets/mulbb7c_mie_pretrain' --valdir '/home/ljm/workspace/datasets/mulbb7c_mie_val'
python NNtrain_arg.py --seed 77 --cuda 'cuda:0' --gama 0.0005 --trainname 'bb7c_abwithPe' --folder 'ablation' --batch 10 --rcsdir '/home/ljm/workspace/datasets/mulbb7c_mie_pretrain' --valdir '/home/ljm/workspace/datasets/mulbb7c_mie_val'