# SleepViTNet

* To train the model:
  * download total 197 subjects' data of sleep-edfx from https://physionet.org/content/sleep-edfx/1.0.0/
  * Put all the .edf data (including annotation files) in `./data`
  * run `extract_data.py` and the extracted .npy files are in `./data/sleep_edf_npy/`
  * run `train.py`
  
* Predict with trained model:
  * the weight in our article is prepared, so you can predict without training
  * run `predict.py`
