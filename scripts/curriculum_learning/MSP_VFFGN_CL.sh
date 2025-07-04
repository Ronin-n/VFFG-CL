set -e
run_idx=$1
gpu=$2

curriculum_stg=('single' 'multiple' 'mix')

for strategy in "${curriculum_stg[@]}"; do
  for i in `seq 1 1 12`;
  do

  cmd="python train_VFFGN.py --dataset_mode=multimodal_miss --model=VFFGN_CL --curriculum_stg=$strategy
  --log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu --image_dir=./shared_image
  --A_type=comparE_raw --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
  --V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
  --L_type=bert_large --input_dim_l=1024 --embd_size_l=128
  --AE_layers=256,128,64 --n_blocks=5 --num_thread=8
  --ce_weight=1.0 --cl_weight=8.0 --ist_weight=100
  --output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
  --niter=38 --niter_decay=2 --verbose --print_freq=10
  --batch_size=16 --lr=5e-4 --run_idx=$run_idx --weight_decay=1e-5 --corpus_name=MSP --Is_part_miss=False --miss_rate=0.3
  --name=MSP_VFFGN_CL --suffix=block_{n_blocks}_run{run_idx} --has_test
  --pretrained_path='checkpoints/MSP_RFFP_AVL_run1'
  --cvNo=$i --num_classes=4"


  echo "\n-------------------------------------------------------------------------------------"
  echo "Execute command: $cmd"
  echo "-------------------------------------------------------------------------------------\n"
  echo $cmd | sh

  done
done
