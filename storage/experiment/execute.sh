id="ailever"
CUDA_VISIBLE_DEVICES=0 python main.py --id $id \
	--epochs 1 \
	--batch_size 200 \
	--dataset_savepath datasets/ \
	--dataset_name MNIST \
	--xlsx_path datasets/dataset.xlsx \
	--json_path datasets/dataset.json \
	--pkl_path datasets/dataset.pkl \
	--hdf5_path datasets/dataset.hdf5 \
	--server http://localhost \
	--port 8097 \
	--env main
