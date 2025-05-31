ml Apptainer

image_dir="/project/tb901114-tb0014/images/candle_image/candle_image.sif"
candle_dir="/project/tb901114-tb0014/images/candle_image/mylib"

export CUDA_COMPUTE_CAP=80
apptainer exec --nv -B $PWD:$PWD -B $candle_dir:/mnt  $image_dir cargo run --release -- --num-shards 2 
ml reset
