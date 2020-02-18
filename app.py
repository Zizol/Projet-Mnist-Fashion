#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from GAN import GAN


# Parsing des arguments d'entrée du script

def parse_args():
    description = "Implémentation Tensorflow d'un GAN"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--epoch', type=int, default=2, help="Le nombre d'epoch à lancer")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist'], help='Le nom du dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='La taille du batch')
    parser.add_argument('--input_size', type=int, default=28, help='Taille de l image d entree')
    parser.add_argument('--lrG', type=float, default=0.0002, help='Learning rate de l optimisateur Adam pour le generateur')
    parser.add_argument('--lrD', type=float, default=0.0002, help='Learning rate de l optimisateur Adam pour le discriminant')
    parser.add_argument('--beta1', type=float, default=0.5, help='Decay rate pour le premier moment des estimations')
    parser.add_argument('--beta2', type=float, default=0.999, help='Decay rate pour le deuxième moment des estimations')
    args = parser.parse_args()

    ## Verification de la conformité des arguments

    assert args.epoch >= 1, 'epoc doit être plus grand ou égal à 1'
    assert args.batch_size >= 1, 'batch size doit être plus grand ou égal à 1'
    assert args.input_size >= 1, 'Taille de l image doit etre plus grand ou égal a 1'
    assert args.lrG <= 1, 'La proportion de poids à être initialement apprise doit être inférieure à 1'
    assert args.lrD <= 1, 'La proportion de poids à être initialement apprise doit être inférieure à 1'
    assert args.beta1 <= 1, 'La baisse pour le premier moment des estimations doit être inférieure à 1'
    assert args.beta2 <= 1, 'La baisse pour le premier moment des estimations doit être inférieure à 1'

    return args



# Script principal

def main():

    # Parsing des arguments
    args = parse_args()

    # Initialisation du generateur
    gan = GAN(args)

        # Lancer l'entrainement
    gan.train()
    print("Entrainement terminé")

        # Visualiser les resultats
    gan.visualize_results(args.epoch)
    print("Tout est fini")

# Lancement du script

if __name__ == '__main__':
    main()


# Données d'entrainement et labels
## Importer datasets


## Afficher quelques images



# Données de validation



# Définition du modèle


# Loss et optimisation


# Metriques


# Entrainement
## Training


## Testing


# Affichage attendu


# Affichage valeurs de test


# Performances



