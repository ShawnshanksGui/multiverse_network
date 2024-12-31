#bash

num_env=1 

fattree_K=16
cc_method=1 #0 = dcqcn, 1 = hpcc(TBD)

cd ./build
make -j 60
cd ..
#python scripts/run.py $_num_env -ds  #--gpu  # > out.file_$_num_port #$num_packet_total #> _${num_packet_total}_${_num_port}.log
python scripts/run.py --num_env $num_env --enable_gpu_sim 'cpu' --fattree_K $fattree_K --cc_method $cc_method
