set -e
run_idx=$1
gpu=$2

curriculum_stg=('single' 'multiple' 'mix')

for strategy in "${curriculum_stg[@]}"; do
  for i in `seq 1 1 10`; do

  cmd="python train_VFFGN.py --dataset_mode=multimodal_miss --model=VFFGN_CL --curriculum_stg=$strategy
  --log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu --image_dir=./shared_image
  --A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
  --V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
  --L_type=bert_large --input_dim_l=1024 --embd_size_l=128
  --AE_layers=256,128,64 --n_blocks=5 --num_thread=8 --corpus=IEMOCAP
  --ce_weight=1.0 --cl_weight=10 --ist_weight=100
  --output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
  --niter=46 --niter_decay=4 --verbose --print_freq=10
  --batch_size=16 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5 --Is_part_miss=False --miss_rate=0.9
  --name=CAP_VFFGN_CL --suffix=block_{n_blocks}_run_{gpu_ids}_{run_idx} --has_test --early_stage_epochs=20
  --pretrained_path='checkpoints/CAP_RFFP_AVL_run1'
  --cvNo=$i --num_classes=4 --random_seed=336 "

  echo "\n-------------------------------------------------------------------------------------"
  echo "Execute command: $cmd"
  echo "-------------------------------------------------------------------------------------\n"
  echo $cmd | sh

  done
done
