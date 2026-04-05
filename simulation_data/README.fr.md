# Données de Simulation

Ce dossier contient les deux jeux de données biologiques qui alimentent la simulation
cerveau-corps de la mouche. Les deux proviennent d'expériences de neurosciences réelles
sur *Drosophila melanogaster* ; les fichiers ne sont pas synthétiques ou approximés,
ils représentent le câblage réellement mesuré du cerveau de la mouche et les réponses
réellement entraînées de ses neurones visuels.

---

## 1. Connectome FlyWire v783 : `flywire_connectome_v783/`

### Qu'est-ce que c'est ?

Une cartographie complète de chaque neurone et chaque connexion synaptique dans le cerveau
adulte de *Drosophila*, reconstruite par le projet FlyWire par microscopie électronique (ME).
Un petit morceau de tissu cérébral de mouche a été découpé en milliers de couches
ultraminces (~40 nm chacune), chaque couche imagée à résolution nanométrique, et le volume
3D résultant assemblé informatiquement. Les neurones ont ensuite été tracés par une
combinaison de segmentation IA et de vérificateurs humains du monde entier. La version 783
est la dernière version publique.

**Échelle :** 138 639 neurones, ~50 millions de synapses.
C'est le premier schéma de câblage quasi-complet d'un cerveau.

**Article :**
> Dorkenwald et al. (2023). *Neuronal wiring diagram of an adult brain.*
> Nature. [doi.org/10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)

**Source (modèle cérébral Shiu et al.) :**
[github.com/philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)

---

### Fichiers

#### `Completeness_783.csv` (~3,3 Mo)

**Ce qu'il contient :** Une ligne par neurone, 138 639 lignes au total.

| Colonne | Contenu |
|---------|---------|
| `root_id` | ID FlyWire du neurone (entier 64 bits unique) |
| `flow` | Classe fonctionnelle large (`sensory`, `ascending`, `descending`, `motor`, `intrinsic`) |
| `super_class` | Classification plus fine (`visual`, `olfactory`, `mechanosensory`, etc.) |
| `cell_class` | Classe anatomique (ex. `LHON`, `MBON`, `DN`) |
| `cell_type` | Label de type cellulaire spécifique quand connu |
| `nt_type` | Neurotransmetteur (glutamate, GABA, acétylcholine, etc.) |

**Comment il est utilisé dans cette simulation :**
C'est l'index principal. Son **ordre de lignes définit l'index des neurones Brian2** :
le neurone à la ligne 0 est le neurone Brian2 0, la ligne 1 est le neurone 1, etc.
Tous les autres fichiers référencent les neurones par cet index de ligne. Nous utilisons
aussi les colonnes `flow` et `super_class` pour identifier les sous-populations :
neurones sensoriels ascendants (1 736), neurones olfactifs (2 279), neurones SEZ/alimentaires
(408), et neurones descendants (1 299) qui pilotent l'asymétrie motrice gauche/droite.

---

#### `Connectivity_783.parquet` (~96 Mo)

**Ce qu'il contient :** Chaque synapse entre chaque paire de neurones connectés, ~50 millions de lignes.

| Colonne | Contenu |
|---------|---------|
| `Presynaptic_Index` | Index de ligne dans Completeness_783.csv du neurone émetteur |
| `Postsynaptic_Index` | Index de ligne du neurone receveur |
| `Excitatory x Connectivity` | Poids synaptique signé (+excitatoire, -inhibiteur) |

Le poids encode à la fois le nombre de contacts synaptiques et le signe du neurotransmetteur.
Une valeur positive signifie que le neurone pré-synaptique excite le neurone post-synaptique ;
une valeur négative signifie une inhibition.

**Comment il est utilisé :**
Cette table est chargée au démarrage pour construire l'objet Brian2 `Synapses` ; la matrice
de connectivité complète de 50 millions de synapses. Chaque ligne devient une synapse Brian2.
Le chargement prend ~30 secondes et est l'étape la plus gourmande en mémoire (~2 Go RAM).
Nous utilisons le format Parquet (binaire en colonnes) plutôt que CSV car il se charge ~10x plus vite.

---

#### `flywire_annotations.tsv` (~31 Mo)

**Ce qu'il contient :** Annotations supplémentaires par neurone de la communauté FlyWire,
une ligne par neurone, liée par `root_id`.

| Colonne | Contenu |
|---------|---------|
| `root_id` | ID FlyWire du neurone (lien avec Completeness_783.csv) |
| `super_class`, `cell_class`, `cell_type` | Labels de types curés par la communauté |
| `side` | Hémisphère : `left`, `right`, ou `center` |
| `soma_x`, `soma_y`, `soma_z` | Coordonnées 3D du corps cellulaire (voxels, résolution 4 nm) |
| `pos_x`, `pos_y` | Position approximative pour les neurones sans coordonnées de soma |

**Comment il est utilisé :**
Les coordonnées de soma (`soma_x`, `soma_y`) positionnent chaque neurone dans le panneau
de visualisation cérébrale. La vue frontale du cerveau dans la vidéo de sortie est construite
en projetant les 138 639 neurones sur un plan 2D grâce à ces coordonnées. Les neurones sans
coordonnées de soma (olfactifs et SEZ) utilisent `pos_x`/`pos_y` en repli. La colonne `side`
sépare les neurones descendants gauches des droits pour le signal d'asymétrie motrice.

---

#### `descending_neurons.csv` (~57 Ko)

**Ce qu'il contient :** Une liste curée de 1 299 neurones descendants (DNs) : neurones qui
portent les commandes du cerveau vers la corde nerveuse ventrale (équivalent de la moelle
épinière chez la mouche) pour contrôler la locomotion.

| Colonne | Contenu |
|---------|---------|
| `root_id` | ID FlyWire du neurone |
| `side` | `left` ou `right` |
| `cell_type` | Sous-type DN (ex. DNa01, DNa02, ...) |
| `brian2_index` | Index de ligne pré-calculé dans la table principale des neurones |

**Comment il est utilisé :**
À chaque pas de simulation de 25 ms, nous comptons combien de DNs gauches ont émis des
spikes vs DNs droits. La différence (`gauche - droite`) produit un petit biais moteur
(multiplié par 0,15) qui ajoute une asymétrie biologiquement fondée à la direction de la
mouche, en complément du gradient olfactif. C'est la seule partie du connectome de
138 639 neurones qui pilote directement le mouvement corporel.

---

### Comment placer les fichiers

La simulation lit ces fichiers depuis `brain_model/` à la racine du projet :

```text
fly_brain_simulation/
  brain_model/
    Completeness_783.csv        <- placer ici
    Connectivity_783.parquet    <- placer ici
    flywire_annotations.tsv     <- placer ici
    descending_neurons.csv      <- placer ici
```

Aucune étape d'installation ; le code Python les lit directement avec `pandas` et `pyarrow`.

---

## 2. Poids Pré-entraînés flyvis : `flyvis_pretrained_weights/`

### Description

Poids synaptiques entraînés pour un **réseau de neurones contraint par le connectome**
du système visuel de la mouche, publié par Lappalainen et al. (2024) dans *Nature*.
L'architecture n'est pas un réseau profond générique ; elle a été conçue pour correspondre
au câblage exact du lobe optique de la mouche, avec 65 types cellulaires arrangés en
une carte rétinotopique hexagonale qui reflète l'anatomie de l'œil composé.

Le réseau a été entraîné pour reproduire les réponses de vrais neurones de mouche mesurées
par imagerie calcique et électrophysiologie, en utilisant la structure du connectome comme
contrainte architecturale stricte. Le résultat est un modèle qui à la fois correspond au
schéma de circuit et prédit l'activité neuronale biologique.

**Types cellulaires (65 au total) :**

- **Photorécepteurs :** R1-R6, R7, R8 : entrée depuis 721 ommatidies par œil
- **Lamine :** L1, L2, L3, L4, L5 : première couche de traitement
- **Médulla :** Mi1, Mi4, Mi9, Tm1, Tm2, Tm3, Tm4, Tm9, ... : extraction de caractéristiques
- **Sélectifs à la direction :** T4a/b/c/d, T5a/b/c/d : détecteurs de mouvement (voies ON et OFF)

**Article :**
> Lappalainen et al. (2024). *Connectome-constrained networks predict neural activity across the fly visual system.*
> Nature. [doi.org/10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)

**Dépôt du code :**
[github.com/TuragaLab/flyvis](https://github.com/TuragaLab/flyvis)

---

### Structure du dossier

```text
flyvis_pretrained_weights/
  0000/                     <- index d'ensemble (un ensemble de modèles entraînés)
    000/                    <- modèle individuel au sein de l'ensemble
      best_chkpt/           <- poids à l'époque avec la meilleure perte de validation
      chkpts/               <- tous les points de contrôle d'entraînement sauvegardés
      validation/           <- métriques de validation
      validation_loss.h5    <- courbe de perte
```

La simulation charge `0000/000` : le premier modèle du premier ensemble :

```python
nv      = NetworkView(flyvis.results_dir / 'flow/0000/000')
network = nv.init_network(checkpoint='best')
```

---

### Ce que les poids encodent

Chaque poids définit la **force synaptique entre deux types cellulaires** à un décalage
hexagonal donné dans la carte rétinotopique. Parce que l'architecture est contrainte par
le connectome, la matrice de poids est creuse ; les connexions qui n'existent pas dans le
vrai lobe optique de la mouche sont structurellement absentes et jamais entraînées.

Le réseau entraîné produit des réponses sélectives à la direction dans T4/T5 :

- **T4** répond aux bords lumineux se déplaçant dans leur direction préférée (voie ON)
- **T5** répond aux bords sombres s'élargissant dans leur champ récepteur (voie OFF)

Un obstacle sombre approchant la mouche produit une région sombre qui s'élargit dans le
champ visuel, exactement le stimulus pour lequel T5 est accordé.

---

### Comment il est utilisé dans cette simulation

À chaque pas de simulation de 25 ms :

1. `obs["vision"]` de MuJoCo donne la luminance brute des ommatidies : forme `(2, 721, 2)`
   (2 yeux x 721 ommatidies x 2 canaux photorécepteurs : jaune/pâle)
2. `RetinaMapper.flygym_to_flyvis()` réordonne les 721 ommatidies de la convention flygym
   vers la convention du réseau hexagonal flyvis (mêmes données, indexation différente)
3. La trame remappée est envoyée au réseau via un seul appel stateful `forward()` (~17 ms)
4. Le réseau conserve son état synaptique récurrent du pas précédent ; pas de remise à zéro
   entre les pas, ce qui est à la fois plus rapide et biologiquement plus correct
5. L'activité T5a et T5b est extraite pour les deux yeux :
   - T5 oeil gauche > T5 oeil droit -> mouvement sombre à gauche -> tourner à droite
   - T5 oeil droit > T5 oeil gauche -> mouvement sombre à droite -> tourner à gauche
6. L'asymétrie T5 devient un biais de virage (limité à +/- 2,0), mélangé au gradient olfactif

C'est la voie d'évitement biologique : oeil composé -> lamine -> médulla -> cellules
T5 sélectives à la direction -> commande de virage. Chez la vraie mouche, cela passe
par les cellules tangentielles de la plaque lobulaire (LPTCs) qui intègrent les signaux
T5 sur tout le champ visuel.

---

### Comment installer

Les poids doivent être placés là où le paquet `flyvis` les attend :

```text
wenv310\Lib\site-packages\flyvis\data\results\flow\0000\
```

**Option A : copier depuis ce dossier** (déjà téléchargés ici) :
Copier `flyvis_pretrained_weights/0000/` dans le chemin ci-dessus.

**Option B : télécharger à nouveau** via la CLI flyvis :

```bash
wenv310\Scripts\flyvis download-pretrained --skip_large_files
```

Le flag `--skip_large_files` ignore les embeddings UMAP et les fichiers d'analyse de
clustering (plusieurs Go) et ne télécharge que les poids du point de contrôle nécessaires
à l'inférence (~6 Mo).
