

def find_feature_axis(z, y, method='linear', **kwargs_model):
    """
    function to find axis in the latent space that is predictive of feature vectors

    :param z: vectors in the latent space, shape=(num_samples, num_latent_vector_dimension)
    :param y: feature vectors, shape=(num_samples, num_features)
    :param method: one of ['linear', 'logistic'], or a sklearn.linear_model object, (eg. sklearn.linear_model.ElasticNet)
    :param kwargs_model: parameters specific to a sklearn.linear_model object, (eg., penalty=’l2’)
    :return: feature vectors, shape = (num_latent_vector_dimension, num_features)
    """

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
