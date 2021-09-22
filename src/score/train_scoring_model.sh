MODEL_DIR=/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/test
WEIGHTS_SAVE_PATH=/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/runs
NAME='lig_clash_score'
DATA_DIR=/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/ml_score/lmdb_data/split/data
cd $MODEL_DIR
#AUGMENT='AUGMENT'
export DATA_DIR=$DATA_DIR
export MODEL_DIR=$MODEL_DIR
export WEIGHTS_SAVE_PATH=$WEIGHTS_SAVE_PATH
TASK='regression'
echo $WEIGHTS_SAVE_PATH
echo $MODEL_DIR

#python train.py --project_name fragment_stitching --run_id $NAME --weights_save_path /
#$WEIGHTS_SAVE_PATH --balance_dataset --num_workers=8 --gpus=1 --batch_size=16 --max_epochs=40 /
# --prefix binding_affinity
python train.py -train $DATA_DIR/train -val $DATA_DIR/val -test $DATA_DIR/test --weights_save_path $WEIGHTS_SAVE_PATH --gpus=1 --num_workers=8 --batch_size=16 --accumulate_grad_batches=2 --learning_rate=0.01 --max_epochs=50 --project_name pdbbind --run_id lig_clash_score --model e3nn --dataset pdbbind --task regression -el HCONF
#commit 7d76223edbb3c9857eb028b875180cf13870132e