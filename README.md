# BCI ALVI Challenge
Code repository to prepare the BCI Initiative + ALVI Labs Challenge


The `utils` folder contains code used to define and train the baseline model as well as to load the data and visualize predictions. The `tutorials` folder includes example code to load the data, train the baseline model and to submit predictions for the competition. 

Please refere to `tutorials/04_submit_predictions.ipynb` for more details on how to prepare the submission file. 

You can access the data on the MUW HPC at the below path (data is stored in the "dataset_v2_blocks/dataset_v2_blocks/"):
/msc/home/vsharm64/projects/BCI_Kaggle


# For installing mamba-ssm it required me to manually install nvcc onto MUW HPC
Add to your .bashrc the following:
```
# Adding cuda driver 12 based on Moritz’s fix
export LD_LIBRARY_PATH=/msc/home/mschae83/NVIDIA-Linux-x86_64-535.129.03/
# Adding nvcc
export PATH=/msc/home/vsharm64/cuda_VS/bin:$PATH
```

Then run
```
source ~/.bashrc
```

After entering your environment (i.e. ALVI) run:
```
pip install --no-build-isolation mamba-ssm
```
