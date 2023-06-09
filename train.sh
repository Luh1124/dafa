# CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size=3 --gpu_ids=0,1,2,3,4,5,6,7 --ext=mainv9 --data_name='vox' --root_dir='/home/momobot/repo/code/2.faceanaimation/dataset/vox1/vox-png'
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv2-1 --root_dir='/home/lh/repo/datasets/vox-png'  --ckp=1
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv2-2-kpc --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv3-0 --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv3-1 --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=2 --gpu_ids=0,1 --ext=mainv7_vox --data_name='vox' --root_dir='/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/' 
# python train.py --batch_size=2 --gpu_ids=0,1 --ext=mainv7_lrw --data_name='lrw' --root_dir='/home/luh/lh_8T/datasets/LRW_Data/LRW_Temp' 
# python train.py --batch_size=8 --gpu_ids=0,1,2,3 --ext=mainv8.2-dl10 --data_name='vox' --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=10 --gpu_ids=0,1,2,3 --ext=mainv9-dl5-lkpc-notanh --data_name='vox' --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=12 --gpu_ids=0,1,2,3 --ext=mainv9finalv1 --data_name='vox' --root_dir='/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/' --display_server=127.0.0.1 --display_port=8098
# CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size=12 --gpu_ids=0,1,2,3,4,5,6,7 --ext=mainv9finalv2 --data_name='vox' --root_dir='../vox-png' --display_server=127.0.0.1 --display_port=8098
# CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size=2 --gpu_ids=0,1 --ext=mainv9finalv1 --data_name='vox' --root_dir='/home/luh/lh_8T/datasets/vox1/face-video-preprocessing/vox-png/' --display_server=127.0.0.1 --display_port=8098
# CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size=3 --gpu_ids=0,1,2,3,4,5,6,7 --ext=mainv9finalv3-lml --data_name='vox' --root_dir='../dataset/vox1/vox-png' --display_server=127.0.0.1 --display_port=8098
# python train.py --batch_size=6 --gpu_ids=0,1,2,3 --ext=mainv9finalv3-lml-dls-newgenmodel --ckp=124 --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=6 --gpu_ids=0,1,2,3 --ext=mainv9finalv3-lml-dls-newgenmodel-mask-largeW --ckp=145 --root_dir='/home/lh/repo/datasets/vox-png' 
# python train.py --batch_size=32 --gpu_ids=0,2,3,4,6 --ext=mainv9finalv3-lml-dls-newgenmodel-mask-slr-sc-lp-ltest2 --ckp=0 --root_dir='/data1/xuhui/data/vox-png' 
# python train.py --batch_size=32 --gpu_ids=2,3,4,5,6 --ext=mainv9finalv3-lml-dls-newgenmodel-mask-slr-sc-lp-ltest2 --ckp=70 --lr=0.00001 --root_dir='/data1/xuhui/data/vox-png' 
# python train.py --batch_size=32 --gpu_ids=0,2,3,4, --ext=mainv9finalv3-lml-dls-newgenmodel-mask-slr-sc-lp-ltest6 --ckp=2 --lr=0.00005 --root_dir='/data1/xuhui/data/vox-png' 

# slr
# python train.py --batch_size=32 --gpu_ids=0,2,3,4, --ext=mainv9finalv3-lml-dls-newgenmodel-mask-slr-sc-lp-ltest6 --ckp=107 --lr=0.00001 --root_dir='/data1/xuhui/data/vox-png' 
# slr large mouth
# python train.py --batch_size=32 --gpu_ids=0,2,3,4, --ext=mainv9finalv3-lml-dls-newgenmodel-mask-slr-sc-lp-ltest6 --ckp=138 --lr=0.00001 --root_dir='/data1/xuhui/data/vox-png' 

python train.py --batch_size=32 --gpu_ids=0,1 --ext=dafa_all --ckp=0 --lr=0.00005 --root_dir='/home/omnisky/repo/datasets/vox1/vox-png'
