

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

    def train_dataloader(self, dataset):
        if dataset == 'mnist':
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../mnist', download=True, train=True,
                                transform=transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    transforms.Lambda(torch.flatten),
                                ])), 
                                batch_size=10000, 
                                shuffle=True)
        elif dataset == 'fashion':
            train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('../fashion', download=True, train=True,
                                transform=transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    transforms.Lambda(torch.flatten),
                                ])), 
                                batch_size=10000, 
                                shuffle=True)
        else:
            raise ValueError("'{}' is not a valid dataset name. Available datasets are: 'mnist' or 'fashion'".format(dataset))
        




class QG_CD_GAN:
    def __init__(self, image_size, batch_size, pca_dims, n_qubits, q_depth, n_generators):
        self.F = Functions()
        self.image_size = image_size
        self.batch_size = batch_size
        self.pca_dims = pca_dims
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators

    def train(self):
            











if __name_ == '__main__':
    pass
