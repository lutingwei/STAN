# GTAN

## Paper data and code

This is the code for the paper entitled "GTAN: Global Target Attention Network for Session-based Recommendation". We have implemented our methods in **PyTorch**.

Here are three benchmark datasets we used in our paper. You can find them on the following websites:

- Diginetica: <http://cikm2016.cs.iupui.edu/cikm-cup>
- Retailrocket: <https://www.kaggle.com/retailrocket/ecommerce-dataset>
- Yoochoose: <http://2015.recsyschallenge.com/challenge.html>

## Usage

The three preprocess files are used to process the raw datasets.

You need to run the file `Preprocess_dn.py` to preprocess the raw Diginetica dataset. `Preprocess_rr.py` corresponds to the Retailrocket dataset and `Preprocess_yo.py` used to preprocess the Yoochoose dataset.

Then you can run the files `dig.py`, `ret.py` or `yoo.py` to train the model.

Evaluation metrics used in the paper are defined in `metric.py`.

## Requirements

- Python 3
- PyTorch 1.1.0
