#!/usr/bin/env python

import json
import os
import shutil
import subprocess

from arima import arima_process
from hashlib import md5

from get_dataset import fetch


def generate_id(id_prefix, context_filepath):
    timestamp = subprocess.check_output(['date', '-u', '+%Y%m%dT%H%M%S.%NZ']).decode().strip()

    with open(context_filepath) as context_file:
        hash_suffix = md5(context_file.read().encode()).hexdigest()[0:5]

    job_id = f'{id_prefix}-{timestamp}-{hash_suffix}'
    print(f'Generated job ID: {job_id}')
    return job_id

pge_root = os.environ['pge_root']
data_root = os.getcwd()
context_filename = os.path.join(data_root, '_context.json')
job_id = generate_id('S1-TIMESERIES-ARIMA', context_filename)
output_root = os.path.join(data_root, job_id)
os.makedirs(output_root)

# TODO: Replace this with localization preprocessor
with open(context_filename) as context_file:
    context = json.load(context_file)
    input_dataset = next(filter(lambda param: param['name'] == 'input_dataset', context['job_specification']['params']))
    url = next(filter(lambda url: url.startswith('s3://'), input_dataset['value']['urls']))
    files = ['timeseries_demErr.h5']
    fetch(url, files, data_root)

filepath = os.path.join(data_root, 'timeseries_demErr.h5')
# ranges for cropping roi (volcano-specific)
yrange, x_range = (1365, 1425), (640, 720)
# index to split train and test series (volcano-specific)
split_idx = 35
# scaling factor for resizing the ROI (downsampling)
scale_factor = 4
# Considers 'n_mse' first mse's as 'normal_mse'
n_mse = 10

# model parameters for each pixel. If not provided, code automatically runs grid search to find the best parameters
# model_params.p contains the values pre-computed for Sierra Negra; this would need to be re-computed for other volcanoes.
model_params = os.path.join(pge_root, 'model_params.p')

arima_process = arima_process(filepath, yrange, x_range, split_idx, scale_factor, n_mse, model_params)
arima_process.train_and_predict()

algo_output_dir = os.path.join(data_root, 'data')
for filename in os.listdir(algo_output_dir):
    shutil.move(os.path.join(algo_output_dir, filename), output_root)

with open(os.path.join(output_root, f'{job_id}.dataset.json'), 'w+') as definition_file:
    json.dump({
        "version": "v1.0",
        # "location": input_dataset['value']['location']
    }, definition_file)

with open(os.path.join(output_root, f'{job_id}.met.json'), 'w+') as metadata_file:
    json.dump({}, metadata_file)