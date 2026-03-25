# Simulation Cerveau-Corps de la Mouche — Documentation Technique

## Vue d'ensemble

Ce projet couple un modèle de réseau de neurones à décharges biologiquement précis du cerveau de *Drosophila melanogaster* (la drosophile, ou mouche du vinaigre) à une simulation physique d'un corps. Le modèle cérébral pilote le corps en temps réel : l'activité neuronale dans le cerveau détermine la vitesse et la direction de marche de la mouche.

**En termes simples :** on simule 138 639 neurones qui s'activent et communiquent entre eux, puis on utilise la sortie de cette simulation pour piloter un modèle physique 3D du corps d'une mouche.

---

## Table des matières

1. [Structure du dépôt](#structure-du-dépôt)
2. [Comment exécuter](#comment-exécuter)
3. [Glossaire biologique](#glossaire-biologique)
4. [Le pipeline complet, étape par étape](#le-pipeline-complet-étape-par-étape)
5. [Approfondissement du modèle cérébral](#approfondissement-du-modèle-cérébral)
6. [Approfondissement du modèle corporel](#approfondissement-du-modèle-corporel)
7. [Interface cerveau-corps](#interface-cerveau-corps)
8. [Paramètres de configuration](#paramètres-de-configuration)
9. [Fichiers de données](#fichiers-de-données)
10. [Dépendances](#dépendances)
11. [Limitations connues et décisions de conception](#limitations-connues-et-décisions-de-conception)
12. [Historique des versions](#historique-des-versions)

---

## Structure du dépôt

```
fly_brain_simulation/
├── brain_body_simulation.py   # v1 : corps flybody, allure tripode codée à la main
├── brain_body_v2.py           # v2 : corps flygym, allure CPG, OdorArena  ← PRINCIPAL
├── main.py                    # Démo d'animation des ailes (standalone)
├── walking_ani.py             # Démo de marche avec politique aléatoire (standalone)
├── set_token.py               # Configuration du jeton API FlyWire
├── .env                       # Config du renderer MuJoCo (MUJOCO_GL=egl)
├── brain_model/
│   ├── model.py               # Constructeur du réseau LIF Brian2
│   ├── utils.py               # Outils d'analyse des trains de décharges
│   ├── Completeness_783.csv   # Index neuronal : 138 639 neurones (FlyWire v783)
│   ├── Connectivity_783.parquet  # Table des synapses : ~50 millions de connexions
│   ├── flywire_annotations.tsv   # Étiquettes de types cellulaires pour tous les neurones
│   └── descending_neurons.csv    # 1 299 neurones de sortie motrice avec étiquettes latérales
└── simulations/
    └── v*.mp4                 # Vidéos de sortie versionnées
```

---

## Comment exécuter

```bash
# Activer l'environnement virtuel
# Windows :
wenv310\Scripts\activate

# Lancer la simulation principale (génère un nouveau .mp4 versionné dans simulations/)
python brain_body_v2.py

# Lancer l'ancienne simulation v1 (flybody, allure tripode)
python brain_body_simulation.py
```

> **Important sous Windows :** `flygym` / `dm_control` doit être importé **avant** l'appel à `load_dotenv()`. Le fichier `.env` définit `MUJOCO_GL=egl`, un renderer GPU spécifique à Linux. S'il est chargé en premier, l'import échoue. L'ordre des imports dans les scripts est intentionnel — ne pas le modifier.

---

## Glossaire biologique

Comprendre ce projet nécessite quelques notions de neurosciences et d'entomologie. Tous les termes sont définis ici dans un langage accessible aux développeurs.

### Drosophila melanogaster
La drosophile, ou mouche du vinaigre. Organisme modèle en biologie — son système nerveux est suffisamment petit pour être entièrement cartographié (~140 000 neurones) mais assez complexe pour produire de vrais comportements : marche, vol, toilettage, navigation olfactive. C'est le "Hello World" de la neurosciences.

### Connectome
Une carte complète de chaque neurone et de chaque synapse (connexion entre neurones) dans un système nerveux. Le connectome de la drosophile a été assemblé à partir d'images de microscopie électronique par le projet FlyWire. La version 783 (utilisée ici) contient **138 639 neurones** et environ **50 millions de synapses**. C'est l'équivalent d'un schéma de câblage complet du cerveau.

### Neurone
L'unité computationnelle de base du système nerveux. Reçoit des entrées électriques, les intègre, et génère une décharge de sortie si l'entrée intégrée dépasse un seuil. Modélisé ici comme une unité **Leaky Integrate-and-Fire (LIF)** (voir ci-dessous).

### Synapse
Une connexion d'un neurone à un autre. Chaque synapse a un **poids** (force) et un **signe** (excitatrice = active la cible, inhibitrice = la supprime). Dans les données du connectome, le signe est encodé par `Excitatory x Connectivity` — les valeurs positives sont excitatrices, les négatives inhibitrices.

### Décharge / Potentiel d'action
Un neurone "s'active" quand sa tension membranaire dépasse un seuil, produisant une brève impulsion électrique appelée décharge (spike). Toutes les informations dans les réseaux neuronaux biologiques sont encodées comme des séquences de ces événements discrets (trains de décharges). Pensez-y comme un signal numérique 1 bit.

### Train de décharges (Spike Train)
La séquence d'horodatages auxquels un neurone a tiré pendant une simulation. Dans Brian2, `spike_monitor.spike_trains()` renvoie un dictionnaire `{id_neurone: tableau_des_temps_de_décharge}`.

### Modèle Leaky Integrate-and-Fire (LIF)
Un modèle mathématique simplifié d'un neurone. La tension membranaire `v` retourne vers un potentiel de repos `v_0` avec le temps (comme un condensateur qui se décharge), mais intègre les décharges entrantes via une variable de conductance `g`. Quand `v` dépasse le seuil `v_th`, le neurone s'active et se réinitialise.

Les équations (tirées de `brain_model/model.py`) :
```
dv/dt = (v_0 - v + g) / t_mbr   [dynamique membranaire]
dg/dt = -g / tau                  [décroissance de la conductance synaptique]
```

Où :
- `v_0 = -52 mV` — potentiel de repos (tension au repos sans entrée)
- `v_th = -45 mV` — seuil (s'active quand v dépasse cette valeur)
- `v_rst = -52 mV` — potentiel de réinitialisation (après activation, v revient ici)
- `t_mbr = 20 ms` — constante de temps membranaire (vitesse de fuite vers le repos)
- `tau = 5 ms` — constante de temps synaptique (vitesse de décroissance d'un spike reçu)
- `t_rfc = 2,2 ms` — période réfractaire (durée minimale entre deux décharges ; le neurone est "sourd" aux entrées pendant cette fenêtre)

### Entrée de Poisson (Poisson Input)
Une façon de modéliser une stimulation externe. Un **processus de Poisson** génère des événements de décharge aléatoires à un taux moyen fixe (ex. : 150 Hz = 150 décharges/seconde en moyenne). Chaque décharge entrante ajoute de la tension à la variable `g` du neurone cible. Cela simule un neurone recevant une entrée bruitée mais soutenue de l'extérieur du réseau modélisé.

### Neurones ascendants
Neurones qui transportent des signaux **du corps vers le cerveau**. Chez la mouche, ils courent physiquement depuis le **Cordon Nerveux Ventral (VNC)** — l'équivalent de la moelle épinière chez la mouche — jusqu'au cerveau. Ils transportent des informations proprioceptives (position des pattes, vitesse de déplacement). Nous stimulons 1 736 de ces neurones comme entrée sensorielle, simulant le signal "les pattes bougent".

> **Pourquoi les neurones ascendants ?** L'analyse de la connectivité montre qu'ils établissent 646 synapses directes sur des neurones descendants liés à la locomotion — bien plus que tout autre type sensoriel. Les stimuler active le plus efficacement le circuit de locomotion.

### Neurones descendants (DNs — Descending Neurons)
Neurones qui transportent des signaux **du cerveau vers le corps**. Ils constituent le seul voie physique entre le cerveau et le système moteur. Chez la mouche, il y a ~1 299 DNs annotés. Ce sont les sorties du cerveau — les "commandes motrices". Nous enregistrons leur activité et l'utilisons pour piloter le corps physique.

DNs notables utilisés pour une cartographie biologiquement fondée :
- **DNa01 / DNa02** — contrôle des pattes avant, initiation et direction de la marche
- **MDN (Moon-walking DN)** — déclenche la marche arrière
- **DNg02** — coordination des pattes médianes et postérieures
- **DNp09 / DNp10** — contrôle des pattes postérieures

### Cordon Nerveux Ventral (VNC — Ventral Nerve Cord)
La "moelle épinière" de la mouche. Situé sous le cerveau, il contient les circuits **Générateurs de Rythmes Centraux (CPG)** qui produisent les mouvements rythmiques des pattes. C'est la pièce manquante de notre simulation — nous le contournons en utilisant le CPGNetwork intégré de flygym. Connecter le vrai VNC du connectome reste un problème de recherche ouvert majeur.

### Générateur de Rythmes Centraux (CPG — Central Pattern Generator)
Un circuit neural qui produit une sortie rythmique sans entrée rythmique. Les CPG existent chez tous les animaux et sont responsables de la marche, de la nage, de la respiration, etc. Ils n'ont pas besoin de retour sensoriel pour fonctionner — ils oscillent d'eux-mêmes. Dans flygym, le CPG est implémenté comme 6 oscillateurs couplés (un par patte) avec des relations de phase qui produisent une allure tripode.

### Allure tripode (Tripod Gait)
Le schéma de marche utilisé par les mouches et de nombreux autres insectes. Les six pattes sont divisées en deux groupes de trois :
- **Tripode A (phase de balancement) :** avant-droite, milieu-gauche, arrière-droite se soulèvent simultanément
- **Tripode B (phase d'appui) :** avant-gauche, milieu-droite, arrière-gauche restent au sol

Les deux tripodes alternent — pendant que A est en l'air, B soutient le corps, et vice versa. Cela maintient toujours au moins 3 pieds au sol, offrant un support stable. C'est l'équivalent insecte du trot diagonal chez les quadrupèdes.

### Proprioception
La sensation de position de ses propres membres dans l'espace. Dans le contexte de la mouche : les pattes envoient des signaux au cerveau signalant leur position et la force exercée. Les neurones ascendants transportent cette information. On la modélise en stimulant ces neurones ascendants avec des entrées de Poisson.

### Olfaction / Taxis olfactif
Le sens de l'odorat. Le **taxis olfactif** est la navigation guidée par des gradients d'odeurs — se déplacer vers les odeurs attrayantes, s'éloigner des odeurs aversives. La mouche détecte les odeurs via ses antennes et ses palpes maxillaires. Dans l'`OdorArena`, la concentration d'odeur suit une loi en inverse du carré (`C ∝ 1/r²`). Les capteurs olfactifs gauche/droite de la mouche lisent la concentration locale, et l'asymétrie résultante guide le virage. Dans notre simulation, le cerveau pilote la locomotion indépendamment — les lectures olfactives ne sont pas encore rebouclées vers le cerveau.

### Potentiel membranaire
La tension électrique à travers la membrane cellulaire d'un neurone. Mesuré en millivolts (mV). Au repos : ~-52 mV. Seuil d'activation : ~-45 mV. Après activation, réinitialisation à : ~-52 mV.

---

## Le pipeline complet, étape par étape

```
[Données du Connectome FlyWire]
        │
        ▼
[1. Construction du réseau LIF Brian2]
   138 639 neurones, ~50M synapses
        │
        ▼
[2. Stimulation des neurones ascendants]
   1 736 neurones, Poisson @ 150 Hz
   (simule le signal "les pattes bougent" du corps vers le cerveau)
        │
        ▼
[3. Simulation cérébrale — 1000 ms]
   ~15 000 neurones s'activent au moins une fois
        │
        ▼
[4. Enregistrement des décharges des neurones descendants]
   645 DNs gauches + 646 DNs droits + 8 DNs bilatéraux
        │
        ▼
[5. Discrétisation des trains de décharges — fenêtres de 25 ms]
   left_rate[40], right_rate[40], both_rate[40]
        │
        ▼
[6. Calcul du signal de contrôle]
   total_amp → vitesse avant
   asymétrie gauche vs droite → biais de virage
   → control_signals[40, 2]
        │
        ▼
[7. Répétition du signal sur 3,75 s]
   ctrl_tiled[150, 2] (signal cérébral répété ~3,75×)
        │
        ▼
[8. Avancement de la physique flygym — 37 500 pas]
   HybridTurningController + CPGNetwork + OdorArena
        │
        ▼
[9. Capture et sauvegarde de la vidéo]
   YawOnlyCamera → vN_brain_body_flygym_odor.mp4
```

---

## Approfondissement du modèle cérébral

### Source
Basé sur [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), qui implémente le réseau de :
> Shiu et al. (2023). A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing. *PLOS Computational Biology.*

### Modèle neuronal (`brain_model/model.py`)

```python
eqs = '''
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
'''
```

Paramètres (de `default_params`) :

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `v_0` | -52 mV | Potentiel de repos |
| `v_rst` | -52 mV | Tension de réinitialisation post-décharge |
| `v_th` | -45 mV | Seuil de décharge |
| `t_mbr` | 20 ms | Constante de temps membranaire |
| `tau` | 5 ms | Décroissance de la conductance synaptique |
| `t_rfc` | 2,2 ms | Période réfractaire |
| `t_dly` | 1,8 ms | Délai de transmission synaptique |
| `w_syn` | 0,275 mV | Poids par synapse |
| `r_poi` | 150 Hz | Taux de stimulation Poisson |
| `f_poi` | 250 | Facteur d'échelle du poids Poisson |

### Connectivité (`Connectivity_783.parquet`)

Chaque ligne est une synapse :

| Colonne | Description |
|---------|-------------|
| `Presynaptic_Index` | Index Brian2 du neurone émetteur |
| `Postsynaptic_Index` | Index Brian2 du neurone récepteur |
| `Excitatory x Connectivity` | Force signée : positif = excitatrice, négatif = inhibitrice |

Le poids synaptique dans Brian2 est : `w = (Excitatory x Connectivity) × w_syn`

### Pourquoi le backend numpy ?
```python
prefs.codegen.target = "numpy"
```
Brian2 compile par défaut vers C++ via Cython pour la vitesse. Cela nécessite un compilateur C++ (Visual Studio sous Windows). Pour éviter cette dépendance, on force le backend numpy pur Python. Il est ~10× plus lent mais ne nécessite aucun compilateur.

### Choix des neurones sensoriels

Nous stimulons les **neurones ascendants** plutôt que les neurones sensoriels (olfactifs, visuels, etc.) parce que :
1. Ils établissent 646 synapses directes sur des DNs de locomotion, atteignant 28 d'entre eux
2. C'est bien plus que tout autre type sensoriel (neurones de l'organe de Johnston : 41 synapses, seulement 3 DNs)
3. Biologiquement, cela simule le retour proprioceptif des pattes en mouvement — une boucle auto-entretenue

---

## Approfondissement du modèle corporel

### Source
[NeuroMechFly v2](https://neuromechfly.org/) (bibliothèque flygym), issu de :
> Lobato-Rios et al. (2023). NeuroMechFly 2.0, a framework for simulating embodied sensorimotor control in adult Drosophila. *Nature Methods.*

### Moteur physique
[MuJoCo](https://mujoco.org/) via [dm_control](https://github.com/google-deepmind/dm_control). MuJoCo signifie **Multi-Joint dynamics with Contact** (dynamique multi-articulaire avec contact). C'est un simulateur physique haute performance développé par DeepMind, standard en robotique et biomécanique.

### Spécifications du modèle de mouche

| Propriété | Valeur |
|-----------|--------|
| Degrés de liberté | 87 articulations |
| Articulations par patte | 7 (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1) |
| Articulations de pattes totales | 42 (6 pattes × 7 DOF) |
| Coussinets adhésifs | Activés (empêche les pieds de glisser) |
| Capteurs olfactifs | Activés (antennes + palpes maxillaires, 4 capteurs au total) |
| Pas de temps physique | 0,1 ms (1e-4 s) |
| Gravité | -9810 mm/s² (gravité standard, modèle en mm) |

### Convention de nommage des articulations

Les pattes sont nommées par position et côté :
- `LF` = Avant Gauche, `LM` = Milieu Gauche, `LH` = Arrière Gauche
- `RF` = Avant Droit, `RM` = Milieu Droit, `RH` = Arrière Droit

Exemple : `LFFemur` = patte avant gauche, articulation du fémur.

L'ordre complet des DOF de pattes (42 valeurs dans le vecteur d'action) :
```
LF: LFCoxa, LFCoxa_roll, LFCoxa_yaw, LFFemur, LFFemur_roll, LFTibia, LFTarsus1
LM: LMCoxa, ...
LH: LHCoxa, ...
RF: RFCoxa, ...
RM: RMCoxa, ...
RH: RHCoxa, ...
```

### HybridTurningController

La classe contrôleur principale. Hérite de `SingleFlySimulation` (un environnement compatible Gymnasium). Reçoit un signal de contrôle 2D à chaque pas :

```python
obs, reward, terminated, truncated, info = sim.step(control_signal)
# control_signal: np.array([amplitude_gauche, amplitude_droite])
# Les deux valeurs dans [0, 1]
```

En interne, il :
1. Définit les amplitudes intrinsèques du CPG depuis `control_signal`
2. Avance 6 oscillateurs couplés (un par patte)
3. Applique la règle de rétraction (soulève les pattes coincées dans des failles)
4. Applique la règle de trébuchement (corrige les pattes qui heurtent des obstacles)
5. Appelle `physics.step()` (1 pas de physique MuJoCo = 0,1 ms)

**Mécanisme de virage :** réduire l'amplitude d'un côté ralentit ces pattes → la mouche courbe vers ce côté.
- `[1.0, 1.0]` → tout droit à pleine vitesse
- `[0.5, 1.0]` → virage à gauche (pattes gauches plus lentes)
- `[1.0, 0.5]` → virage à droite (pattes droites plus lentes)
- `[0.3, 0.3]` → marche lente (les deux côtés au minimum)

### CPGNetwork

6 oscillateurs, un par patte, avec des **biais de phase tripode**. La matrice de relations de phase assure que le Tripode A (LF, RM, RH) et le Tripode B (RF, LM, LH) oscillent en opposition de phase (π radians d'écart). Fréquence par défaut : 12 Hz (12 cycles de pas complets par seconde).

### OdorArena

Un plan de sol plat avec des sources d'odeur ponctuelles. Concentration d'odeur à la position `r` d'une source : `C(r) = intensité_max / r²` (diffusion en inverse du carré). Les 4 capteurs olfactifs de la mouche (2 antennes + 2 palpes maxillaires) lisent chacun la concentration locale. L'asymétrie gauche/droite des capteurs peut guider les virages — c'est la base biologique de la navigation olfactive.

Dans notre simulation : la source d'odeur existe dans l'arène mais le modèle cérébral ne la lit pas encore. C'est l'infrastructure pour une intégration future.

---

## Interface cerveau-corps

C'est le cœur du projet. La réalité biologique est que les neurones descendants (DNs) sont les **seuls câbles** reliant le cerveau au système moteur. On lit leur activité et on la traduit en une commande de locomotion 2D.

### Étape 1 : Discrétisation des trains de décharges

La simulation cérébrale de 1000 ms est divisée en 40 fenêtres de 25 ms chacune. Pour chaque fenêtre, on compte combien de décharges ont été émises par chaque groupe de DNs :

```python
bin_edges = np.linspace(0.0, 1.0, 41)   # 40 fenêtres
left_rate[t]  = somme des décharges dans la fenêtre t des DNs côté gauche
right_rate[t] = somme des décharges dans la fenêtre t des DNs côté droit
both_rate[t]  = somme des décharges dans la fenêtre t des DNs bilatéraux
```

### Étape 2 : Propulsion en avant (amplitude)

Activité totale des DNs → vitesse de marche :
```python
total_rate = left_rate + right_rate + both_rate
# Lisser avec une moyenne mobile (fenêtre = 8 bins = 200 ms)
total_smooth = convolve(total_rate, fenêtre_uniforme)
# Normaliser dans [0, 1]
dn_amp = total_smooth / total_smooth.max()
```

### Étape 3 : Biais de virage (asymétrie)

Différence de tir gauche vs droite des DNs → direction de virage :
```python
lr_diff = (left_rate - right_rate) / (left_rate + right_rate + ε)
# Plage : [-1, 1]
# Positif → plus de DNs gauches → la mouche tourne à gauche
# Négatif → plus de DNs droits → la mouche tourne à droite
```

### Étape 4 : Signal de contrôle

```python
base = MIN_AMP + (1 - MIN_AMP) * dn_amp[t]   # [0,3, 1,0]
bias = lr_diff[t] * 0.4                        # ±0,4 maximum

control_signals[t, 0] = clip(base - bias, 0, 1)  # pattes gauches
control_signals[t, 1] = clip(base + bias, 0, 1)  # pattes droites
```

`MIN_AMP = 0.3` garantit que la mouche ne s'arrête jamais complètement — biologiquement, une mouche recevant un retour proprioceptif de ses pattes en mouvement devrait continuer à marcher.

### Étape 5 : Répétition temporelle

Le cerveau tourne pendant 1 seconde, mais on veut 3,75 secondes de physique (= 15 secondes de vidéo à 0,25× de vitesse de lecture). Le signal de contrôle de 40 décisions est répété (~3,75×) pour remplir la durée complète de la simulation.

---

## Paramètres de configuration

Toutes les constantes réglables sont en haut de `brain_body_v2.py` :

| Paramètre | Défaut | Effet |
|-----------|--------|-------|
| `BRAIN_DURATION_MS` | 1000 | Durée de la simulation neuronale (ms). Plus long = plus de diversité dans les patterns de contrôle, mais plus de temps de calcul. |
| `STIM_RATE_HZ` | 150 | Taux de stimulation Poisson des neurones ascendants (Hz). Plus élevé = plus de neurones s'activent = signal de locomotion plus fort. |
| `DECISION_INTERVAL` | 0,025 | Durée de chaque fenêtre de contrôle (secondes). Plus court = plus réactif aux fluctuations neuronales. |
| `PHYSICS_TIMESTEP` | 1e-4 | Taille du pas de physique MuJoCo (secondes). Ne pas dépasser 1e-4 ou la simulation devient instable. |
| `MIN_AMP` | 0,3 | Amplitude minimale du CPG (0–1). Empêche la mouche de s'arrêter quand l'activité des DNs est faible. |
| `VIDEO_DURATION_S` | 15,0 | Durée cible de la vidéo de sortie (secondes) à `play_speed`. |
| `play_speed` | 0,25 | Vitesse de lecture vidéo par rapport au temps réel. 0,25 = ralenti × 4. La physique tourne 0,25× aussi longtemps que la vidéo. |

---

## Fichiers de données

### `brain_model/Completeness_783.csv`
Table d'index neuronal du connectome FlyWire version 783.

| Colonne | Description |
|---------|-------------|
| Index (label de ligne) | ID racine FlyWire (entier 64 bits unique à chaque neurone) |
| Autres colonnes | Métadonnées de complétude (couverture axone/dendrite) |

L'ordre des lignes définit l'index neuronal Brian2. Le neurone `i` dans Brian2 correspond à la ligne `i` de ce CSV.

### `brain_model/Connectivity_783.parquet`
Table des synapses. Chaque ligne est une synapse (connexion directionnelle).

| Colonne | Description |
|---------|-------------|
| `Presynaptic_Index` | Index Brian2 du neurone émetteur |
| `Postsynaptic_Index` | Index Brian2 du neurone récepteur |
| `Excitatory x Connectivity` | Force signée : positif = excitatrice, négatif = inhibitrice |

### `brain_model/flywire_annotations.tsv`
Annotations de types cellulaires pour les 139 244 neurones.

Colonnes clés :
- `root_id` — ID FlyWire (correspond à l'index CSV)
- `super_class` — type grossier : `ascending`, `descending`, `visual`, `olfactory`, `central`, etc.
- `cell_class` — classification plus fine
- `cell_type` — type nommé spécifique (ex. : `MDN`, `DNa01`, `DNg02_a`)
- `side` — `left`, `right`, ou vide (bilatéral)

Source : Schlegel et al. (2024), données supplémentaires de l'article FlyWire.

### `brain_model/descending_neurons.csv`
Sous-ensemble filtré : 1 299 neurones descendants.

| Colonne | Description |
|---------|-------------|
| `root_id` | ID FlyWire |
| `cell_type` | Type nommé (ex. : `MDN`, `DNa01`) |
| `top_nt` | Neurotransmetteur dominant |
| `side` | `left`, `right`, ou vide |

---

## Dépendances

Toutes installées dans `wenv310/` (environnement virtuel Python 3.10, Windows) :

| Paquet | Rôle |
|--------|------|
| `brian2` | Simulation de réseau de neurones à décharges |
| `flygym` (v1.2.1) | Modèle corporel NeuroMechFly + environnement MuJoCo |
| `dm_control` | Bindings Python MuJoCo de DeepMind |
| `mujoco` | Moteur physique |
| `numpy` | Calcul matriciel |
| `pandas` | Chargement des données du connectome |
| `python-dotenv` | Chargement du `.env` pour la config du renderer |
| `imageio` | Encodage vidéo (utilisé par le Camera de flygym) |
| `ffmpeg` | Codec vidéo (dépendance système, appelé par imageio) |

---

## Limitations connues et décisions de conception

### 1. Le VNC est contourné
Le vrai cerveau de la mouche se connecte au VNC qui contient les CPGs produisant la marche. Nous avons remplacé le VNC par le `CPGNetwork` de flygym. Le cerveau fournit seulement un signal 2D grossier de vitesse/direction — il ne génère pas les commandes individuelles pour chaque articulation. Combler cet écart nécessiterait un modèle complet du connectome du VNC.

### 2. Le signal cérébral est répété en boucle
Le cerveau tourne pendant 1 seconde et produit 40 fenêtres de contrôle. Pour une vidéo de 15 secondes, ce signal est répété ~3,75×. La mouche marche donc avec le même "rythme" neural en boucle. Un correctif approprié consisterait à faire tourner le cerveau en continu en parallèle de la physique, avec un retour sensoriel fermant la boucle.

### 3. Pas de boucle de retour sensoriel
Les neurones ascendants sont stimulés avec du bruit de Poisson fixe — ils ne répondent pas réellement aux mouvements des pattes simulées. Un système en boucle fermée lirait les positions des pattes depuis la simulation physique et les utiliserait pour définir l'activité des neurones ascendants. C'est l'écart clé entre ce travail et le résultat d'Eon Systems.

### 4. L'olfaction n'est pas connectée au cerveau
L'`OdorArena` envoie des lectures d'odeur aux capteurs virtuels de la mouche, mais ces données ne sont pas rebouclées dans le modèle Brian2. Connecter les neurones olfactifs (neurones de projection dans le lobe antennaire) au modèle cérébral permettrait une vraie navigation guidée par les odeurs.

### 5. Sortie neuronale stochastique
Brian2 avec entrée de Poisson est stochastique — chaque exécution produit des trains de décharges légèrement différents à cause du processus de Poisson aléatoire. C'est biologiquement réaliste (les neurones sont bruités) mais signifie que la mouche marchera légèrement différemment à chaque exécution.

### 6. Ordre des imports (spécifique à Windows)
L'ordre `flygym` → `load_dotenv()` doit être respecté. Voir [Comment exécuter](#comment-exécuter).

---

## Historique des versions

| Fichier | Vidéo(s) | Description |
|---------|----------|-------------|
| `main.py` | v1 | Animation des ailes, pas d'avancement physique |
| `walking_ani.py` | v2 | Marche avec politique aléatoire via l'env `walk_imitation` (inclut une trajectoire fantôme de référence) |
| `brain_body_simulation.py` | v3–v5 | Corps flybody, allure tripode codée à la main, entrée de neurones ascendants, readout moteur DN |
| `brain_body_v2.py` | v6+ | NeuroMechFly v2 flygym, HybridTurningController + CPGNetwork, OdorArena, YawOnlyCamera |

### Sous-versions de `brain_body_v2.py` (suivies par numéro de vidéo)

| Vidéo | Changement clé |
|-------|---------------|
| v6 | Première exécution flygym, caméra fixe vue du dessus |
| v7 | Étendu à 15 s, caméra fixe |
| v8 | YawOnlyCamera `camera_top_right` — caméra de suivi isométrique |
| v9 | Ajout d'une phase de toilettage (pattes avant levées vers le visage) |
| v10 | Tentative `camera_right_front` + angles de toilettage plus agressifs → mouche renversée |
| v11+ | Retour à la configuration stable v8, sans toilettage |
