

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

    def relu(self, x):
        return x * (x > 0)

    def get_noise_upper_bound(self, gen_loss, disc_loss, original_ratio):
        R = disc_loss.detach().numpy()/gen_loss.detach().numpy()
        return math.pi/8 + (5 *math.pi / 8) * relu(np.tanh((R - (original_ratio))))

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
        

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(pca_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class PatchQuantumGenerator(nn.Module):
    def __init__(self, n_generators, device, n_qubits=5, n_a_qubits=1, q_depth=6, n_generators=4, q_delta=1):
        """ n_generators (int): Number of sub-generators
            n_qubits (int): Total number of qubits / N
            n_a_qubits (int): Number of ancillary qubits / N_A
            q_depth (int): Depth of the parameterised quantum circuit / D
            device (string): Device; quantum device
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators
        self.device = device
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @qml.qnode(self.device, diff_method="parameter-shift")
    def quantum_circuit(self, noise, weights):
        weights = weights.reshape(q_depth, n_qubits)
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)
        # For each layer
        for i in range(q_depth):
            # RY Gates
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)
            # Control Z gates
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
        return qml.probs(wires=list(range(n_qubits)))

    def partial_measure(self, noise, weights):
        probs = quantum_circuit(noise, weights)
        probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
        probsgiven0 /= torch.sum(probs)

        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven

    def forward(self, x):
        patch_size = 2 ** (n_qubits - n_a_qubits)
        images = torch.Tensor(x.size(0), 0).to(self.dev)
        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(self.dev)
            for elem in x:
                q_out = self.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            images = torch.cat((images, patches), 1)
        return images


class QG_CD_GAN:
    def __init__(self, dataset, image_size, pca_dims, n_qubits, q_depth, n_generators):
        self.F = Functions()
        self.dataset = dataset
        self.image_size = image_size
        self.pca_dims = pca_dims
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators

    def train(self, dev, lrG, lrD, num_iter):
        train_data = self.F.train_dataloader(dataset='')
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gen_losses = []
        disc_losses = []
        discriminator = Discriminator().to(device)
        generator = PatchQuantumGenerator(n_generators).to(device)
        criterion = nn.BCELoss()
        optD = optim.SGD(discriminator.parameters(), lr=lrD)
        optG = optim.SGD(generator.parameters(), lr=lrG)
        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)
        counter = 0
        noise_upper_bound = math.pi/8
        fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

        results = []

        while True:
            for i, (data, label) in enumerate(train_loader):
                # Data for training the discriminator
                # data = data.reshape(-1, image_size * image_size)
                real_data = data.to(device)

                # Noise follwing a uniform distribution in range [0,pi/2)
                noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
                fake_data = generator(noise).reshape(batch_size, 1, 8, 8)

                # Training the discriminator
                discriminator.zero_grad()
                outD_real = discriminator(real_data).view(-1)
                outD_fake = discriminator(fake_data.detach()).view(-1)

                errD_real = criterion(outD_real, real_labels)
                errD_fake = criterion(outD_fake, fake_labels)
                # Propagate gradients
                errD_real.backward()
                errD_fake.backward()

                errD = errD_real + errD_fake
                optD.step()

                # Training the generator
                generator.zero_grad()
                outD_fake = discriminator(fake_data).view(-1)
                errG = criterion(outD_fake, fake_labels)
                errG.backward()
                optG.step()
        
                counter += 1

                disc_losses.append(errD.cpu())
                gen_losses.append(errG.cpu())
        
                # Show loss values
                if counter % 10 == 0:
                    print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
        
                    test_embeddings = generator(fixed_noise).view(1,image_size,image_size).cpu().detach()
                    if counter % 50 == 0:
                        results.append(test_embeddings)
        
                if counter == num_iter:
                    break
            if counter == num_iter:
                break




if __name_ == '__main__':
    pass
