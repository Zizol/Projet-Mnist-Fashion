# Projet-Deep Learning
Etude des gan avec le [dataset MNIST Fashion de Zalando]((https://github.com/zalandoresearch/fashion-mnist))

## Abstract

Les réseaux de neurones génératif connaissent un certain succès pour leur applications ludiques, mais aussi utilitaires,
car les jeux de données labellisés demandent du temps pour être construits.
Nous nous sommes intéressés à quelques modèles : tout d'abord la structure GAN, puis **une autre**.
Nous avons pour cela utilisé les données MNIST, puis le jeu de données plus récent Fashion-MNIST de Zalando.
Le résultat consiste à générer des chiffres et des vetements grâce à ces réseaux.


## Introduction et Etat de l'art

Depuis le premier neurone artificiel de McCulloch et Pitt
en 1943, les réseaux sont devenus très performants grâce
notamment à l’utilisation des GP-GPU et des gros jeux de
données. En 1985, le français Y. Le Cun met en place les
réseaux multi-couches et la rétro-propagation du gradient.

Ce système fut utilisé à la fin des années 90 pour la reconnaissance de chiffres sur les chèques bancaires.

Dès les années 80, les auto-encodeurs permettent d’obtenir une méthode intéressante non supervisée grâce à une
reconstruction de l’entrée avec compression intermédiaire.
En 2012, un deep CNN (Convolutional Neural Network)
a dépassé les performances des autres solutions de reconnaissance sur ImageNet (ILSVRC) de presque 10%, passant de 25% 
d’erreur à 16%. Depuis, les performances des
réseaux de neurones supervisés ne cessent de croître, alors
que les méthodes non supervisées ont plus de mal à s’améliorer.

En 2014, l’arrivée des GANs apportent alors des méthodes de génération d’images intéressantes (non supervisée). 
Les DCGANs (Deep Convolutional GANs) [4] et les
InfoGANs sont développés avec des applications très diverses : génération de chiffres manuscrits (entraînés Mnist),
de visages ou de chambres à coucher.

Les GANs sont un sujet de recherche très actif. Il
existe encore plusieurs problèmes, notamment en ce qui
concerne la convergence. Les WGANs (Wasserstein GANs) tentent de palier à ce problème en remplaçant la divergence de 
Jensen-Shannon, la fonction de coût classique des
GANs, par la distance de Wasserstein qui est corrélée à la
qualité de l’image et améliore alors la convergence.

## Approche
### Jeux de données
#### MNIST

Afin de tester nos réseaux génératifs, nous avons dans un
premier temps utilisé le jeu de données de chiffres manuscrits MNIST. Ce jeu de données présente l’avantage d’être
relativement riche (60 000 exemples d’entraînement et 10
000 exemples de tests) et d’être très bien documenté car
il a été utilisé pour de nombreux travaux. C’est souvent le
jeu de données utilisé dans les premiers exemples de Deep
Learning.

Les scores de l’état de l’art aujourd’hui s’approchent de 99,8% 2 de reconnaissance.

### Principe de fonctionnement d’un GAN

Un GAN comporte un réseau génératif G qui génère des
données à partir d’un bruit d’entrée et un réseau discriminant D qui cherche à distinguer les vraies données (c’est
à dire celles issues du jeu de données) des «fausses» données, c’est-à-dire celles générées par G. Ainsi G joue le rôle
de faussaire et cherche à tromper D, tandis que ce dernier
joue le rôle de la police qui cherche à détecter les données
contrefaites. Les deux réseaux s’entraînent ainsi mutuellement. Au fur et à mesure des itérations, G génère des objets
de plus en plus proches de ce qu’on peut trouver dans le
jeu de données (par exemple, des images qui ressemblent
de plus en plus à des chiffres). De même, D distingue de
mieux en mieux les faux exemples générés par G.
L’article publié en 2014 par l’université de Montréal et
notamment écrit par Ian J. Goodfellow explique les mécanismes mis en jeu par les GANs. Il se focalise sur le
cas particulier où le réseau génératif G est un MLP (MultiLayer Perceptron) aux entrées duquel on applique un bruit
aléatoire, et où le réseau discriminant D est aussi un MLP.

Ces deux réseaux peuvent être entraînés avec la backpropagation. Les données générées par G utilisent quant à
eux uniquement la «forward propagation». On entraîne D
à maximiser

Cependant, lorsque D rejette les échantillons avec une forte probabilité (ce qui a des chances de se produire notamment 
en début d’entraînement),
log(1 − D(G(z))) sature et on obtient un gradient insuffisant. En pratique on entraîne donc plutôt G à minimiser −log(D(G(z)))

L’utilisation de ces réseaux de neurones présente plusieurs avantages en comparaison par rapport à d’autres approches 
génératives plus classiques, notamment l’absence
d’utilisation de chaînes de Markov : les gradients sont obtenus à partir de la backpropagation uniquement. De plus,
aucune inférence n’est nécessaire pendant l’apprentissage
et une large variété de fonctions peuvent être incorporées
au modèle.

Cependant, il n’y a pas de représentation explicite de la probabilité p(x) du réseau génératif G et le réseau
discriminant D doit être bien synchronisé avec D pendant
l’apprentissage.

## Experimentations

### Difficulté d’évaluation des algorithmes génératifs

Pour les algorithmes d’apprentissage supervisé, il existe
de nombreuses méthodes d’évaluation de leurs performances. Par exemple pour un classifieur binaire on peut
utiliser la courbe ROC, la mesure F ou simplement le taux
d’erreur sur un set de test différent du training set. En revanche pour les algorithmes génératifs l’évaluation de la
qualité de la génération est moins directe.
Un article de Deepmind mets en garde contre
la non-équivalence entre les différentes méthodes : log-
vraisemblance moyenne, estimation de la fenêtre de Parzen
et fidélité visuelle d’échantillons. La non-équivalence entre
les différentes mesure implique que l’on ne peut pas facilement obtenir le «meilleur» générateur, car son score dépend
de la mesure choisie. Ce choix doit se faire en fonction des
applications prévues du réseau.
Cependant, une fois la mesure performance du réseau
choisie, il est possible de sélectionner les hyperparamètres
avec une cross validation sur l’ensemble de training divisé
en plusieurs couples training + validation. C’est d’ailleurs
ce qui est présenté dans l’article fondateur des GANs de
Goodfellow.

De même, régulariser ses réseaux et éviter l’overfitting,
Goodfellow utilise du dropout pour les réseaux discriminants de ses GANs.

### MNIST avec GANs

Afin de nous familiariser avec les GAN nous avons utilisé un premier exemple implémenté pour pytorch pour générer des 
chiffres manuscrits à partir de MNIST. On donne au discriminant simultanément une vraie image du dataset catégorisée 
comme telle, et une image crée par le générateur à partir de bruit. Le discriminant calcule alors le score associé aux 
prédiction d'authenticité de ces deux images.

Il utilise pour le générateur une couche Relu suivie d’une couche tanh. Il prend en entrée un vecteur de bruit uniforme de -1
à 1 et de dimension 62. La sortie a une taille
784 = 28 × 28 (soit la taille d’un échantillon de MNIST). Pour le discriminant il s'agit d'une couche leakyRelu puis Sigmoid

L’entraînement d’un Infogan donne pour les premières
itérations des images très bruités (ce qui est normal car les
échantillons sont générés à partir d’un bruit). Cependant dès
la 23 000ème itération (sur 1 million avec des batchs de
taille 128). On commence à obtenir des images qui ont des
allures de chiffres manuscrits.
Après 400 000 itérations, le générateur produit des échantillons comme ceux illustrés dans la figure 15. La plupart des 
sorties générées à ce moment pourraient tout à fait être reconnues par un humain comme des lettres manuscrites. Cependant 
on constate encore que certaines sorties
comme celle en bas à droite (sorte de mélange de 8 et de
9) sont encore éloignées de la représentation d’un chiffre
manuscrit.

### Fashion-MNIST avec GAN

Grosso modo pareil, a un moment il y a un pic qui chamboule tout c'est impressionant

### MNIST avec InfoGAN

On va bien voir, en gros ca prend un vesteur en entrée qui dit quel chiffre il doit générer. Encore plus fort, on peut 
lui en donner deux pour générer le meme chiffre sous différents angles

## Conclusions

 Ce projet a été pour nous l’occasion de nous confronter
à la réalité de l’état de l’art sur les deux types de GANs
les plus utilisés à l’heure actuelle. La prise en main des
auto-encodeur et GANs nous ont permis de réaliser que
des structures relativement simples permettent de générer
des échantillons assez convaincants. Mais que la convergence de ces réseaux est difficile à obtenir : une des difficultés 
rencontrées a été le temps de calcul d’entraînement
des GANs, les résultats obtenus sur les sets de lettres manuscrites pourraient probablement être améliorés avec un
plus grand nombre d’itérations.
Une piste intéressante d’expérimentation sur les GANs
serait le fine tuning (des dernières couches des réseaux génératifs ou discriminants), cela permettrait (peut-être) de 
réduire les temps d’entraînement ou de diminuer le nombre
d’images nécessaires pour spécialiser les neurones sur des
tâches ou des sets spécifiques. On pourrait également essayer d’entraîner ces réseaux avec un plus grand nombre de
lettres (il suffirait pour cela d’augmenter la taille des entrées) ou d’entraîner plusieurs réseaux sur moins de lettres.
Cela permettrait peut-être de réduire temps d’entraînement
de chaque réseau et d’avoir des réseaux spécialisés dans la
génération de certaines lettres (ce systèmes serait peut-être
plus performants) Les GANs sont très en vogues, les
approches supervisées ont énormément progressé pendant
les 5 dernières années notamment grâce à leur réussite dans
les concours de reconnaissances. Les approches non supervisées ou semisupervisées sont cependant très importantes
car les sets de données labellisées sont coûteux et encore
peu nombreux (par rapport aux données non labellisés).


## References

https://github.com/znxlwm/pytorch-generative-model-collections

https://reiinakano.com/gan-playground/

https://arxiv.org/abs/1606.03657