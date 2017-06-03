import constants as c
import utils
import numpy as np
from sklearn.preprocessing import StandardScaler

features_min_max = {}
X_scaler = None

def store_features_min_max(hists, sbins, hogs):
    global features_min_max
    hists_concat = np.concatenate(hists, axis=0)
    mins = np.min(np.min(hists_concat,axis=0), axis=2)
    maxs = np.max(np.max(hists_concat,axis=0), axis=2)
    mins_maxs = {'min':mins, 'max':maxs}
    features_min_max[c.hists]=mins_maxs

    sbins_concat = np.concatenate(sbins, axis=0)
    mins = np.min(np.min(sbins_concat,axis=0), axis=1)
    maxs = np.max(np.max(sbins_concat,axis=0), axis=1)
    mins_maxs = {'min':mins, 'max':maxs}
    features_min_max[c.sbins]=mins_maxs

    hogs_concat = np.concatenate(hogs, axis=0)
    mins = np.min(np.min(hogs_concat,axis=0), axis=2)
    maxs = np.max(np.max(hogs_concat,axis=0), axis=2)
    mins_maxs = {'min':mins, 'max':maxs}
    features_min_max[c.hogs]=mins_maxs


def get_features_min_max(feature_type, color_space, channel=None):
    f_min = features_min_max[feature_type]['min'][color_space]
    f_max = features_min_max[feature_type]['max'][color_space]
    if channel is not None:
        f_min = f_min[channel]
        f_max = f_max[channel]
    return f_min, f_max

def get_X_scaler():
    global X_scaler
    return X_scaler


def combine_features(hists, sbins, hogs, sel_hists=[], sel_sbins=[], sel_hogs=False):
    features=np.zeros((hists.shape[0],1), dtype=np.float32)       # "empty" features to concat to

    for cspace, channel in sel_hists:
        f = hists[:, cspace, channel]
        features = np.concatenate((features, f),axis=1)

    for cspace in sel_sbins:
        f = sbins[:, cspace]
        features = np.concatenate((features, f),axis=1)

    for cspace, channel in sel_hogs:
        f = hogs[:, cspace, channel]
        features = np.concatenate((features, f),axis=1)

    features = features[:,1:]  # remove 0 column that was created for placeholder
    return features

def get_features(use_features):
    """read feature pickles and generate X,y based on use_features"""

    vehicle_hists = utils.unpickle_data(c.vehicles_histograms_p)
    non_vehicle_hists = utils.unpickle_data(c.non_vehicles_histograms_p)
    print('vehicle hists: ',vehicle_hists.shape)
    print('non_vehicle hists: ',non_vehicle_hists.shape)

    vehicle_sbins = utils.unpickle_data(c.vehicles_spatial_bins_p)
    non_vehicle_sbins = utils.unpickle_data(c.non_vehicles_spatial_bins_p)
    print('vehicle bins: ',vehicle_sbins.shape)
    print('non_vehicle bins: ',non_vehicle_sbins.shape)

    vehicle_hogs = utils.unpickle_data(c.vehicles_hog_p)
    non_vehicle_hogs = utils.unpickle_data(c.non_vehicles_hog_p)
    print('vehicle hog: ',vehicle_hogs.shape)
    print('non_vehicle hog: ',non_vehicle_hogs.shape)

    # store_features_min_max((vehicle_hists, non_vehicle_hists),(vehicle_sbins, non_vehicle_sbins),(vehicle_hogs, non_vehicle_hogs))

    vehicle_features = combine_features(vehicle_hists, vehicle_sbins, vehicle_hogs,
                                        use_features[c.hists], use_features[c.sbins], use_features[c.hogs])

    non_vehicle_features = combine_features(non_vehicle_hists, non_vehicle_sbins, non_vehicle_hogs,
                                            use_features[c.hists], use_features[c.sbins], use_features[c.hogs])

    X = np.concatenate((vehicle_features,non_vehicle_features))
    y = np.concatenate((np.ones(vehicle_features.shape[0]), np.zeros(non_vehicle_features.shape[0])))

    global X_scaler
    X_scaler = StandardScaler().fit(X)
    X = X_scaler.transform(X)

    return X, y, X_scaler


if __name__=='__main__':
    use_features = {
        c.hists: [[c.hls_index, 2],
                #   [c.xyz_index, 0],
                #   [c.xyz_index, 1],
                #   [c.xyz_index, 2],
                  [c.luv_index, 0],
                  [c.luv_index, 1],
                  [c.luv_index, 2]],
        c.sbins: [c.hls_index,
                  c.xyz_index,
                  c.luv_index],
        c.hogs: [[c.xyz_index, 0]]
                #  [c.luv_index, 1],
                #  [c.luv_index, 2]]
    }

    X, y = get_features(use_features)

    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    import time

    print('X:', X.shape)
    print('y:', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # svc = LinearSVC(True)
    svc = SVC(probability=True)
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

