from arima import arima_process


if __name__ == '__main__':

	filepath = '/Users/cendon/Desktop/Volcano Anomaly/volcano_data/aria-data/galapagos_201412-202007-1/timeseries_demErr.h5'
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
	model_params = 'model_params.p'

	arima_process = arima_process(filepath, yrange, x_range, split_idx, scale_factor, n_mse, model_params)
	arima_process.train_and_predict()