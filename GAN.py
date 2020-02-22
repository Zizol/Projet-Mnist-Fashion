# -*- coding: utf-8 -*-

import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Modèle du générateur
class generator(nn.Module):

    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size


        # Couche fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )

        # Couche convolutionnelle
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    # L'input avance dans le réseau
    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


# Modèle du discriminant
class discriminator(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        # Couche convolutionnelle
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Couche fully connected
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    # L'input avance dans le réseau
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


# Le modèle GAN
class GAN(object):
    def __init__(self, args):

        # Parametres
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.z_dim = 62
        self.result_dir = 'results'
        self.save_dir = 'models'
        self.dataset = args.dataset
        self.model_name = 'GAN'

        # Chargement des données
        self.data_loader = utils.dataloader(self.input_size, self.batch_size, self.dataset)
        data = self.data_loader.__iter__().__next__()[0]

        # Initialisation du générateur et du discriminant
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)

        # Optimisation avec Adam
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.G.cpu()
        self.D.cpu()
        self.BCE_loss = nn.BCELoss().cpu()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')


        # Bruit
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        self.sample_z_ = self.sample_z_.cpu()


    # Entrainement du réseau
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        # 1 = Vrai, 0 = Faux
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_, self.y_fake_ = self.y_real_.cpu(), self.y_fake_.cpu()

        # Début de l'entrainement du discriminant
        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):

            # Pour chaque epoch, le générateur est entrainé
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):

                # Pour sortir de la boucle en cas de dépassement d'index
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                x_, z_ = x_.cpu(), z_.cpu()

                # Mets à jour le réseau D
                self.D_optimizer.zero_grad()

                # Le discriminant prend une vraie image et calcule la loss de sa prédiction
                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                # Le générateur génère à partir de bruit, le discriminant calcule la loss de cette prédiction
                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                # La loss totale est l'aggrégation de la loss pour la vrai image et pour le bruit
                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                # La loss est rétro propagée
                D_loss.backward()
                self.D_optimizer.step()

                # Mets à jour les poids de G
                self.G_optimizer.zero_grad()

                # Le générateur calcule la loss de sa génération à partir de la prédiction du discriminant
                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                # La loss est rétro propagée
                G_loss.backward()
                self.G_optimizer.step()

                # Tout les 100 iterations affiche l'avancée du programme
                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
                    with torch.no_grad():
                        self.visualize_results((epoch + 1), (iter + 1))
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)


        # Affiche des données sur le temps d'execution
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        # Enregistre les résultats et crée un graphe d'évolution de la loss
        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    # Visualisation des résultats
    def visualize_results(self, epoch, iteration, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        # Génération à partir du bruit (fixe ou non)
        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            sample_z_ = sample_z_.cpu()

            samples = self.G(sample_z_)

        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

        # Sauvegarde
        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '_iter%03d' % iteration + '.png')

    # Utilitaire pour sauvegarder les résultats de l'entrainement
    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
