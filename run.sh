
num_env=1 
num_updates=100000

gpu_id=3
fattree_K=4
cc_method=1 #0 = dcqcn, 1 = hpcc(TBD)

cd ./build
cmake ..
make -j 60
cd ..
python scripts/train.py --num-worlds $num_env --num-updates $num_updates --ckpt-dir build/ckpts  --gpu_id $gpu_id --fattree_K $fattree_K --cc_method $cc_method  #--gpu-sim  
