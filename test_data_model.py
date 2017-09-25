import numpy as np

from data_model_jg import StockDataSet

input_size=30
overlap=1./6.
num_steps=30
k=None
target_symbol='AMZN'
test_ratio=0.05

a = StockDataSet(target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio, normalized=False, overlap=overlap)
b = StockDataSet(target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio, normalized=True, overlap=overlap)

print(a.train_X.shape)