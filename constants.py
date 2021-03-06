
train_data_folder = 'data'
vehicles_train_data_folder = train_data_folder + '/vehicles'
non_vehicles_train_data_folder = train_data_folder + '/non-vehicles'

non_vehicles_auto_folder = train_data_folder + '/non-vehicles/auto'

video_images_folder = 'videos'
project_video_images_folder = video_images_folder + '/project'
rain_video_images_folder = video_images_folder + '/rain'
test_video_images_folder = video_images_folder + '/test'

vehicles_train_data_p = train_data_folder + '/vehicles.z'
non_vehicles_train_data_p = train_data_folder + '/non_vehicles.z'
vehicles_preprocessed_train_data_p = train_data_folder + '/vehicles_pre.z'
non_vehicles_preprocessed_train_data_p = train_data_folder + '/non_vehicles_pre.z'
vehicles_histograms_p = train_data_folder + '/vehicles_hist.z'
non_vehicles_histograms_p = train_data_folder + '/non_vehicles_hist.z'
vehicles_spatial_bins_p = train_data_folder + '/vehicles_sbin.z'
non_vehicles_spatial_bins_p = train_data_folder + '/non_vehicles_sbin.z'
vehicles_hog_p = train_data_folder + '/vehicles_hog.z'
non_vehicles_hog_p = train_data_folder + '/non_vehicles_hog.z'
svm_p = train_data_folder + '/svm.z'
x_scaler_p = train_data_folder + '/x_scaler.z'
test_detections_p = train_data_folder + '/test_detections.z'
all_detections_p = train_data_folder + '/all_detections.z'
dl_detections_p = train_data_folder + '/dl_detections.z'


output_folder = 'output_images'
project_video = 'project_video.mp4'
test_video = 'test_video.mp4'

output_folder_dl = output_folder + '/dl'
output_folder_dl_rain = output_folder + '/dl_rain'

# sample size
sample_size = 64

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

