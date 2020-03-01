# Projet-Deep Learning
Etude des gan avec le [dataset MNIST Fashion de Zalando]((https://github.com/zalandoresearch/fashion-mnist))

## Lancer un modèle

On peut lancer un modèle avec `python app.py`. L'application lancera de base un GAN sur MNIST pour 2 epoch.

On peut modifier les paramètres, notamment :
* `--gan_type = ['GAN', 'InfoGAN', 'WGAN']`
* `--epoch = int`
* `--dataset = ['mnist', 'fashion-mnist']`

## Les données générées

Les images associées aux modèles et aux datasets se trouvent dans `/result`.

Les courbes de pertes se trouvent dans `/models`