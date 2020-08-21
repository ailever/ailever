id="ailever"
localhost_ip="222.99.138.172"
port="8100"

python -m visdom.server -p $port &
if [ ! -z $localhost_ip ]
then
	ssh -N -f -L localhost:$port:localhost:$port dongmyeong@$localhost_ip &
fi
CUDA_VISIBLE_DEVICES=0 python main.py --id $id \
	--epochs 1 \
	--batch_size 200 \
	--dataset_savepath datasets/ \
	--dataset_name CIFAR100 \
	--xlsx_path datasets/dataset.xlsx \
	--json_path datasets/dataset.json \
	--pkl_path datasets/dataset.pkl \
	--hdf5_path datasets/dataset.hdf5 \
	--server http://localhost \
	--port $port \
	--env main
