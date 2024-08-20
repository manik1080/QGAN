

class Functions:
    def __init__(self):
        pass
    
    def scale_data(data, scale=None, dtype=np.float32):
        if scale is None:
            scale = [-1, 1]
        min_data, max_data = [float(np.min(data)), float(np.max(data))]
        min_scale, max_scale = [float(scale[0]), float(scale[1])]
        data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
        return data.astype(dtype)
    
    def frechet_dist(self, act1, act2):
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def principal_component(self, ncomp, data):
        pca = PCA(n_components=ncomp)
        pca_data = pca.fit_transform(data)

    def get_dataloader(self, )


class QG_CD_GAN(Functions):
    def __init__(self):
        pass
    def 
