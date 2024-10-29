import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
import math
from torch import nn
import pennylane as qml
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt

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
        return pca, pca_data
    
    def reverse_pca(self, pca, pca_data):
        return pca.inverse_transform(pca_data)

    def relu(self, x):
        return x * (x > 0)

    def get_noise_upper_bound(self, gen_loss, disc_loss, original_ratio):
        R = disc_loss.detach().numpy()/gen_loss.detach().numpy()
        return math.pi/8 + (5 *math.pi / 8) * relu(np.tanh((R - (original_ratio))))

    def train_dataloader(self, dataset, batch_size=8, pca_components=None):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        if dataset == 'mnist':
            dataset = datasets.MNIST('../mnist', download=True, train=True,
                                transform=transforms.Compose([
                                    # torchvision.transforms.Resize(shape),
                                    torchvision.transforms.ToTensor(),
                                    transforms.Lambda(torch.flatten),
                                ]))
        elif dataset == 'fashion':
            dataset = datasets.FashionMNIST('../fashion', download=True, train=True,
                                transform=transforms.Compose([
                                    # torchvision.transforms.Resize(shape),
                                    torchvision.transforms.ToTensor(),
                                    transforms.Lambda(torch.flatten),
                                ]))
        else:
            raise ValueError("'{}' is not a valid dataset name. Available datasets are: 'mnist' or 'fashion'".format(dataset))
        if pca_components:    
            data_numpy = dataset.data.view(len(dataset), -1).numpy() / 255.0
            pca, pca_data = self.principal_component(pca_components, data_numpy)
            pca_tensor_data = torch.from_numpy(pca_data).float()
            train_loader = DataLoader(pca_tensor_data, batch_size=batch_size, shuffle=True)
            return train_loader, pca
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class PatchQuantumGenerator(nn.Module):
    def __init__(self, device, n_generators=4, n_qubits=5, n_a_qubits=1, q_depth=6, q_delta=1):
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
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators
        self.device = device
        self.q_delta = q_delta
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        @qml.qnode(device, diff_method="parameter-shift")
        def quantum_circuit(noise, weights):
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

        def partial_measure(noise, weights):
            probs = self.quantum_circuit(noise, weights)
            probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
            probsgiven0 /= torch.sum(probs)

            probsgiven = probsgiven0 / torch.max(probsgiven0)
            return probsgiven
        
        self.partial_measure = partial_measure
        self.quantum_circuit = quantum_circuit

    def forward(self, x):
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits)
        images = torch.Tensor(x.size(0), 0).to(self.dev)
        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(self.dev)
            for elem in x:
                q_out = self.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            images = torch.cat((images, patches), 1)
        return images


class QG_CD_GAN:
    def __init__(self, dataset, image_size, pca_components, n_qubits=5, n_a_qubits=1, q_depth=6, q_delta=1, n_generators=4):
        self.F = Functions()
        self.dataset = dataset
        self.image_size = image_size
        self.feature_shape = pca_components if pca_components else image_size * image_size
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators
        self.pca_components = pca_components
        self.n_a_qubits = n_a_qubits
        self.q_delta = q_delta
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fixed_noise = torch.rand(1, self.n_qubits, device=self.device) * math.pi / 2

    def train(self, dev, lrG, lrD, num_iter, batch_size=1):
        if self.pca_components:
            train_loader, pca = self.F.train_dataloader(dataset=self.dataset, pca_components=self.pca_components)
        else:
            train_loader = self.F.train_dataloader(dataset=self.dataset)
        
        gen_losses = []
        disc_losses = []
        discriminator = Discriminator(self.feature_shape).to(self.device)
        generator = PatchQuantumGenerator(dev, n_generators=self.n_generators, n_qubits=self.n_qubits, n_a_qubits=self.n_a_qubits, q_depth=self.q_depth, q_delta=self.q_delta).to(self.device)
        criterion = nn.BCELoss()
        optD = optim.SGD(discriminator.parameters(), lr=lrD)
        optG = optim.SGD(generator.parameters(), lr=lrG)
        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=self.device)
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=self.device)
        counter = 0
        noise_upper_bound = math.pi/8
        noise = torch.rand(batch_size, self.n_qubits, device=self.device) * math.pi / 2

        results = []
        test_embeddings = []

        while True:
            try:
                for i, data in enumerate(train_loader):
                    real_data = data.to(self.device)
                    noise = torch.rand(batch_size, self.n_qubits, device=self.device) * math.pi / 2
                    fake_data = generator(noise)#.reshape(batch_size, 1, 8, 8)

                    # Training the discriminator
                    discriminator.zero_grad()
                    outD_real = discriminator(real_data).view(-1)
                    outD_fake = discriminator(fake_data.detach()).view(-1)

                    errD_real = criterion(outD_real, real_labels)
                    errD_fake = criterion(outD_fake, fake_labels)

                    errD = errD_real + errD_fake
                    errD.backward()
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
            
                        if self.pca_components:
                            test_embeddings.append(self.F.reverse_pca(pca, generator(self.fixed_noise).cpu().detach().numpy()).reshape(self.image_size, self.image_size))
                        else:
                            test_embeddings.append(generator(self.fixed_noise).cpu().detach().numpy().reshape(self.image_size, self.image_size))

                        if counter % 100 == 0:
                            results.extend(test_embeddings)
                            test_embeddings = []
            
                    if counter == num_iter:
                        break
                if counter == num_iter:
                    break
            except KeyboardInterrupt:
                print('Interrupted')
                break
            
        return {'gen_losses': gen_losses, 'disc_losses': disc_losses, 'results': results,
                'generator': generator, 'discriminator': discriminator, 'num_iter': num_iter,
                'pca': pca if self.pca_components else None, 'train_loader': train_loader}


if __name__ == '__main__':
    f = Functions()
    d = Discriminator(10)
    g = PatchQuantumGenerator(n_generators=4, n_qubits=5, n_a_qubits=1, q_depth=6, q_delta=1, device=qml.device("lightning.qubit", wires=5))
    model = QG_CD_GAN(dataset='mnist', image_size=28, pca_components=64, n_qubits=5, q_depth=6, n_generators=4, n_a_qubits=1, q_delta=1)
    # train_loader = f.train_dataloader(dataset='mnist')
    print(type(train_loader))
    print(train_loader.dataset[0])
    plt.imshow(train_loader.dataset[0][0].reshape((8,8)), cmap='gray')
    plt.show()
    print(d)
    print(g)
    res = model.train(dev=qml.device("lightning.qubit", wires=5),
            lrG=0.3,
            lrD=0.05,
            num_iter=1000,
            batch_size=8)
    # plotting result
    result = res['results']
    for i in range(len(result)):
        plt.subplot(5, 6, i+1)
        plt.imshow(result[i][0], cmap='gray')
        plt.axis('off')
    plt.show()

    noise = (torch.rand(1, 5, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) * math.pi / 2)
    image = f.reverse_pca(res['pca'], res['generator'](noise).cpu().detach().numpy()).reshape(28, 28)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()
    
    train_loader[0][0].shape
    # save model
    torch.save(res['generator'].state_dict(), 'generator.pth')
    torch.save(res['discriminator'].state_dict(), 'discriminator.pth')
