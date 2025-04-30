# Projet TP – TP Traitement Automatique d’Image et Vidéo

**Préparé par :**  
Oussama BOUSSAHLA – Souhil Mokeddem

---

## Table des Matières

1. [Introduction](#introduction)  
2. [Fonctionnalités Principales](#fonctionnalités-principales)  
3. [Installation](#installation)  
4. [Exécution](#exécution)  
5. [Structure du Code](#structure-du-code)  
   1. [Utilitaires Partagés](#utilitaires-partagés)  
   2. [TP1 – Histogrammes](#tp1--histogrammes)  
   3. [TP2 – Transformations](#tp2--transformations)  
   4. [TP3 – Traitement des Couleurs](#tp3--traitement-des-couleurs)  
   5. [TP4 – Convolution & Filtrage](#tp4--convolution--filtrage)  
   6. [TP5 – Contours](#tp5--contours)  
   7. [TP6 – Segmentation](#tp6--segmentation)  
   8. [TP7 – Étiquetage CC (Connected Components)](#tp7--étiquetage-cc-connected-components)  
   9. [TP8 – Propriétés de Régions & Codes de Freeman](#tp8--propriétés-de-régions--codes-de-freeman)  
   10. [TP9 – Traitement Vidéo “From Scratch”](#tp9--traitement-vidéo-from-scratch)  
6. [Dépendances](#dépendances)  
7. [Améliorations Futures](#améliorations-futures)  

---

## Introduction

Ce projet est une application **Streamlit** interactive pour l’enseignement et l’expérimentation des principales étapes du traitement d’images et vidéos “from scratch”.  
Il couvre :

- Opérations sur histogrammes  
- Transformations d’intensité (translation, inversion, étirement, égalisation, spécification)  
- Traitement de la couleur et quantification  
- Convolution / filtrage  
- Détection de contours  
- Segmentation par seuillage et classification  
- Étiquetage de composantes connexes  
- Calcul de propriétés de régions et codes de Freeman  
- Extraction et visualisation de frames vidéo avec **FFmpeg**  

Chaque section (“TP”) est accessible depuis la barre latérale de l’application.

---

## Fonctionnalités Principales

- Chargement d’images (PNG, JPEG, BMP, TIFF)  
- Histogrammes :  
  - Classique, normalisé, cumulatif, cumulatif normalisé  
- Transformations d’intensité :  
  - Translation de luminosité (slider ±100)  
  - Inversion (négatif)  
  - Dynamic Range Expansion (étirement)  
  - Égalisation (OpenCV & manuelle)  
  - Spécification d’histogramme à partir d’une image de référence  
- Traitement couleur :  
  - Conversion RGB ⇄ HSV / LAB  
  - Égalisation couleur  
  - Quantification : uniforme, K-Means, median-cut  
- Convolution & filtrage :  
  - Convolution 2D scratch (moyenne, gaussien, laplacien, kernel custom)  
  - Filtre médian scratch  
- Contours :  
  - Détection “maison” Canny & Sobel  
  - Filtrage par aire minimale  
  - Overlay des contours + statistiques (aire, périmètre, centroïde)  
- Segmentation :  
  - Seuillage (manuel, Otsu, Haris)  
  - Classification K-Means sur niveaux de gris  
- Étiquetage CC :  
  - Binarisation + connected components + coloration  
- Propriétés de régions & codes de Freeman :  
  - Aire, périmètre, centroïde  
  - Code de Freeman (direction + symboles)  
  - Téléchargement CSV  
- Vidéo “from scratch” :  
  - Upload vidéo, extraction de frames via FFmpeg, affichage par slider  

---

## Installation

1. Cloner le dépôt  
   ```bash
   git clone https://votre-repo.git
   cd tp-image-video
