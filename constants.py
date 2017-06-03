
train_data_folder = 'data'
vehicles_train_data_folder = train_data_folder + '/vehicles'
non_vehicles_train_data_folder = train_data_folder + '/non-vehicles'

video_images_folder = 'videos'
project_video_images_folder = video_images_folder + '/project'
test_video_images_folder = video_images_folder + '/test'

vehicles_train_data_p = train_data_folder + '/vehicles.z'
non_vehicles_train_data_p = train_data_folder + '/non_vehicles.z'
vehicles_histograms_p = train_data_folder + '/vehicles_hist.z'
non_vehicles_histograms_p = train_data_folder + '/non_vehicles_hist.z'
vehicles_spatial_bins_p = train_data_folder + '/vehicles_sbin.z'
non_vehicles_spatial_bins_p = train_data_folder + '/non_vehicles_sbin.z'
vehicles_hog_p = train_data_folder + '/vehicles_hog.z'
non_vehicles_hog_p = train_data_folder + '/non_vehicles_hog.z'
svm_p = train_data_folder + '/svm.z'
x_scaler_p = train_data_folder + '/x_scaler.z'


# sample size
sample_w = 64
sample_h = 64
sample_size = (sample_w, sample_h)

# hog parameters
hog_orient=8
hog_pix_per_cell=8
hog_cell_per_block=2
hog_transform_sqrt = True


# index of color spaces in multispace arrays
bgr_index = 0
hls_index = 1
xyz_index = 2
luv_index = 3

# feature names
hists = 'hists'
sbins = 'sbins'
hogs='hogs'

