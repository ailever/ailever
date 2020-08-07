id="ailever"
CUDA_VISIBLE_DEVICES=0 python main.py --id $id \
	--epochs 5 \
	--batch_size 8 \
	--xlsx_path datasets/dataset.xlsx \
	--json_path datasets/dataset.json \
	--pkl_path datasets/dataset.pkl \
	--hdf5_path datasets/dataset.hdf5
