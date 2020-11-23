import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings

from skimage.transform import resize, downscale_local_mean
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from eval_arima import *
from arima_viz_utils import *

class arima_process():
    
    def __init__(self, filepath, yrange, xrange, split_idx, scale_factor, n_mse, model_param_path=None):
        
        if not os.path.exists('data'):
            os.makedirs('data')
            
        datafile = h5py.File(filepath, 'r')
        ts_allregion = np.array(datafile['timeseries'])

        self.dates = np.array([d.decode("utf-8") for d in datafile['date']])
        self.timeseries = self.extract_volcanodata(ts_allregion, yrange, xrange, add_bias=True)
        
        self.n_mse = n_mse
        self.scale_factor = scale_factor
        self.split_idx = split_idx

        def preprocess_data():
            self.plot_2d(self.timeseries[-1, :, :], "Example 2D frame")
            self.resize_timeseries(scale_factor)
            self.split_traintest()
            self.viz_split_traintest()
            if (model_param_path is None):
                self.find_arima_parameters(self.train)
            else:
                self.load_arima_parameters(model_param_path)
        
        preprocess_data()
        
    def split_traintest(self):
        self.train = self.timeseries_downscl[:self.split_idx]
        self.test = self.timeseries_downscl[self.split_idx:]
    
    def resize_timeseries(self, scale_factor):
        nt, ysize, xsize = self.timeseries.shape
        yshp, xshp = downscale_local_mean(self.timeseries[-1, :, :], (scale_factor, scale_factor)).shape
        self.timeseries_downscl = np.zeros((nt, yshp, xshp))
        for i, fr in enumerate(self.timeseries):
            self.timeseries_downscl[i] = downscale_local_mean(self.timeseries[i, :, :], (scale_factor, scale_factor))
        self.plot_2d(self.timeseries_downscl[-1, :, :], "Example 2d Downscaled")
        
    def find_arima_parameters(self, train):
        p_values = range(0, 3)
        d_values = range(0, 3)
        q_values = range(0, 3)
        warnings.filterwarnings("ignore")
        
        nt, ysize, xsize = self.timeseries_downscl.shape
        train_data = train.reshape(-1, ysize*xsize) # flattened training data
        self.model_params = []
        for pix in range(ysize*xsize):
            print ('Evaluating pixel ', pix)
            best_params = evaluate_models(train_data[:, pix], p_values, d_values, q_values)
            self.model_params.append(best_params)
            
        pickle.dump(self.model_params, open(os.path.join('data', "model_params_gs.p"), "wb" ))
        
        return
    
    def load_arima_parameters(self, model_param_path):
        self.model_params = pickle.load(open(model_param_path, 'rb'))
    

    def plot_2d(self, data, title):
        plt.figure()
        plt.title(title)
        plt.imshow(data, cmap='jet')
        plt.savefig(os.path.join('data', f'{title}.png'))
        
    def add_bias_to_data(self, data):
        data += abs(np.nanmin(data))
        return data
    
    def extract_volcanodata(self, timeseries, y_range, x_range, add_bias=False):
        ysize = y_range[1]-y_range[0]
        xsize = x_range[1]-x_range[0]
        nt = timeseries.shape[0]
        data = np.zeros((nt, ysize, xsize))
        for i, fr in enumerate(timeseries):
            data[i] = fr[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        if (add_bias):
            data = self.add_bias_to_data(data)
        return data
    
    def get_ticks_and_labels(self, val_range, n_ticks, start_date_idx):  
        xticks_pos = np.linspace(val_range[0], val_range[1], n_ticks, dtype='int')
        xticks_labels = [self.dates[i+start_date_idx] for i in xticks_pos]
        return xticks_pos, xticks_labels
    
    def viz_split_traintest(self):
        ysize, xsize = self.timeseries_downscl.shape[1:]
        ysize_all, xsize_all = self.timeseries.shape[1:] # time series approx center
        sample_1d_all = self.timeseries[:, int(ysize_all/2), int(xsize_all/2)]
        sample_1d_train = np.full(self.timeseries_downscl.shape[0], np.nan)
        sample_1d_train[:self.split_idx] = self.train[:, int(ysize/2), int(xsize/2)] # time series approx center for viz
        sample_1d_test = np.full(self.timeseries_downscl.shape[0], np.nan)
        sample_1d_test[self.split_idx:] = self.test[:, int(ysize/2), int(xsize/2)]

        plt.figure()
        data_1d = sample_1d_all
        plt.plot(data_1d, 'b.')
        plt.ylabel('Displacement')
        plt.xlabel('Time Sample')
        xticks_pos, xticks_labels = self.get_ticks_and_labels([0, len(sample_1d_all)-1], 5, 0)
        plt.xticks(xticks_pos, xticks_labels)
        plt.title('All Data')
        plt.savefig(os.path.join('data','alldata_example.png'))

        plt.figure()
        plt.title('Train and Test Series')
        plt.plot(sample_1d_train, 'b.', sample_1d_test, 'r.')
        plt.ylabel('Displacement')
        plt.xlabel('Time Sample')
        xticks_pos, xticks_labels = self.get_ticks_and_labels([0, len(sample_1d_all)-1], 5, 0)
        plt.xticks(xticks_pos, xticks_labels)
        plt.legend(['Train', 'Test'])
        plt.savefig(os.path.join('data', 'traintest_example.png'))
        
    def evaluate(self, observed, pred):
        try:
            return abs(mean_squared_error(observed, pred))
        except:
            return np.nan

    def fit_arima(self, data, order):
        warnings.filterwarnings("ignore")
        
        model = ARIMA(data, order=order)
        
        try:
            model_fit = model.fit(disp=0)
        except:
            model_fit = np.nan
            
        return model_fit
    
    def train_and_predict(self):
        
        assert (self.train.shape[1:] == self.timeseries_downscl.shape[1:])
        assert (self.test.shape[1:] == self.timeseries_downscl.shape[1:])
        
        nt, ysize, xsize = self.train.shape
        self.test_dates = self.dates[self.split_idx:]
        
        # Datastructures
        train_data = self.train.reshape(-1, ysize*xsize) # flattened time series
        test_data = self.test.reshape(-1, ysize*xsize)   # flattened time series
        
        anomaly_map = np.zeros((test_data.shape)) # stores mse of the entire region
        pred_map = np.zeros((test_data.shape)) # store predicted values entire region

        update_model = np.array([True]*ysize*xsize) # boolean array indicates which pixel needs model update
        self.normal_mse = [np.nan for _ in range(ysize*xsize)] # stores normal_mse. 

        self.arima_models = dict() # saves current arima model for each pixel

        for i, date in enumerate(self.test_dates): # Going over test datapoints sequentially
            print ('Processing date : ', date)
            
            for pix in range(ysize*xsize): # Looping over every pixel on current frame

                if (update_model[pix]): # Re-trains the arima model if True
                    if (i == 0):
                        data_train_update = train_data[:, pix]
                    else:
                        data_train_update = np.concatenate((train_data[:, pix], test_data[:i-1, pix])) # appends test data before current date
  
                    new_model = self.fit_arima(data_train_update, order=self.model_params[pix])
                    if new_model is np.nan:
                        # Find better parameters
                        p_values = range(0, 4)
                        d_values = range(0, 3)
                        q_values = range(0, 3)
                        self.model_params[pix] = evaluate_models(data_train_update, p_values, d_values, q_values)
                        new_model = self.fit_arima(data_train_update, order=self.model_params[pix])
                    
                    if (new_model is not np.nan): # update model only if not nan, otherwise keep old one
                        self.arima_models[pix] = new_model 
                        n_remain_dates = self.test_dates.shape[0]-i
                        pred_map[i:, pix] = self.arima_models[pix].forecast(n_remain_dates)[0] # updates predictions; predicts 'n_remain_dates' values, i.e, all values until the end of the test sequence

                # Get mse between prediction and observed value   
                anomaly_map[i, pix] = self.evaluate([test_data[i, pix]], [pred_map[i, pix]])

                if (anomaly_map[i, pix] is np.nan): # update the model if mse is nan
                    update_model[pix] = True 
                elif (i >= self.n_mse):
                    if self.normal_mse[pix] is np.nan:
                        self.normal_mse[pix] = self.find_normal_mse(anomaly_map[:, pix], i) 
                    update_model[pix] = self.decide_modelupdate(self.normal_mse[pix], anomaly_map[i, pix])
                else:
                    normal_mse = np.nanmean(anomaly_map[:i, pix])
                    update_model[pix] = self.decide_modelupdate(normal_mse, anomaly_map[i, pix])

        # Reshaping and Saving Output
        self.anomaly_map = self.get_fullmap(anomaly_map, os.path.join('data', "full_anomaly_map.p"))
        #self.pred_map = self.get_fullmap(pred_map, os.path.join('data', "full_pred_map.p"))
        self.pred_map = self.get_fullmap(pred_map)
        self.obs_map = self.get_fullmap(self.test, os.path.join('data', "full_obs_map.p"))
        
        self.save_output_plots()
        
    def find_normal_mse(self, mse_hist, i):
        n = self.n_mse
        normal_mse = np.nanmean(mse_hist[:n])
        while((normal_mse is np.nan) and (n < i)):
            n += 1
            normal_mse = np.nanmean(mse_hist[:n])
            
        return normal_mse

    def get_fullmap(self, data, path=None):
        # Reshape anomaly map
        map_rs = np.zeros((data.shape[0], self.test.shape[1], self.test.shape[2]))
        for i, fr in enumerate(data):
            map_small = fr.reshape((self.timeseries_downscl.shape[1:])) # reshape flattened data into 2d structure
            map_rs[i] = resize(map_small, self.test.shape[1:], anti_aliasing=True)
        
        if (path is not None):
            pickle.dump({'data' : map_rs, 'dates' : self.test_dates}, open(path, "wb"))
        return map_rs
        
    def decide_modelupdate(self, normal_mse, mse):
        if normal_mse is np.nan:
            return True

        if (mse > 10*normal_mse):
            return True
        else:
            return False
        
    def save_2dmap(self, data, date, path):
        amap = data.reshape((self.timeseries_downscl.shape[1:]))
        amap_upscl = resize(amap, self.timeseries.shape[1:], anti_aliasing=False)
        
        plt.figure()
        plt.title(date)
        plt.imshow(amap_upscl, cmap='jet', vmin=0, vmax=0.5)
        plt.savefig(path)
        
    def save_output_plots(self):
        #plot_timeseries2d(self.pred_map, self.test_dates, "Prediction Maps",
        #                  os.path.join('data', 'prediction_maps'))
        plot_timeseries2d(self.anomaly_map, self.test_dates, "Anomaly Maps", 
                          os.path.join('data', 'anomaly_maps'))
        plot_timeseries1d(self.pred_map, self.obs_map, self.test_dates, 
                          6, 6, "Prediction vs. Real Values in 1D",
                         'data')
        