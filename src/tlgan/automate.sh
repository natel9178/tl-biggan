python feature_axis.py -labels_loc=./gan_samples_labels.hdf5 -samples_loc=./gan_samples.hdf5 -f_save_dir=./feature_direction_dogs.pkl -features_limit=337
python feature_axis.py -labels_loc=./gan_samples_cats_labels.hdf5 -samples_loc=./gan_samples_cats.hdf5 -f_save_dir=./feature_direction_cats.pkl -features_limit=337
python feature_axis.py -labels_loc=./gan_samples_laptop_labels.hdf5 -samples_loc=./gan_samples_laptop.hdf5 -f_save_dir=./feature_direction_laptop.pkl -features_limit=337
python feature_axis.py -labels_loc=./gan_samples_washingmachine_labels.hdf5 -samples_loc=./gan_samples_washingmachine.hdf5 -f_save_dir=./feature_direction_washingmachine.pkl -features_limit=337
python feature_axis.py -labels_loc=./gan_samples_pineapple_labels.hdf5 -samples_loc=./gan_samples_pineapple.hdf5 -f_save_dir=./feature_direction_pineapple.pkl -features_limit=337
python feature_axis.py -labels_loc=./gan_samples_figs_labels.hdf5 -samples_loc=./gan_samples_figs.hdf5 -f_save_dir=./feature_direction_figs.pkl -features_limit=337