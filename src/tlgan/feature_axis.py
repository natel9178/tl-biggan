import argparse
import numpy as np
import sklearn.linear_model as linear_model
import h5py
import pickle

def find_feature_axis(z, y, method='linear', **kwargs_model):
    if method == 'linear':
        model = linear_model.LinearRegression(**kwargs_model)
        model.fit(z, y)
    elif method == 'tanh':
        def arctanh_clip(y):
            return np.arctanh(np.clip(y, np.tanh(-3), np.tanh(3)))

        model = linear_model.LinearRegression(**kwargs_model)

        model.fit(z, arctanh_clip(y))
    else:
        raise Exception('method has to be one of ["linear", "tanh"]')

    return model.coef_.transpose()

def normalize_feature_axis(feature_slope):
    feature_direction = feature_slope / np.linalg.norm(feature_slope, ord=2, axis=0, keepdims=True)
    return feature_direction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-labels_loc', default='./tlgan/gan_samples_labels.hdf5')
    parser.add_argument('-samples_loc', default='./tlgan/gan_samples.hdf5')
    parser.add_argument('-f_save_dir', default='./feature_direction.pkl')
    opt = parser.parse_args()

    with h5py.File(opt.labels_loc, 'r') as l, h5py.File(opt.samples_loc, 'r') as s:
        z, y = l['latent_vector'][:], s['labels'][:]
        feature_slope = normalize_feature_axis(find_feature_axis(z, y, method='tanh'))

        with open(opt.f_save_dir, 'wb') as f:
            pickle.dump(feature_slope, f)

if __name__ == "__main__":
    main()