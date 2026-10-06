# AllskyStudio

Application de bureau (Linux, customtkinter) pour traiter les nuits de la
caméra allsky [Multicam](https://github.com/remis-astr/Multicam) : images
`.jpg`, `.png`, `.fit` ou `.fits` (RAW Bayer ou mono).

## Fonctions

- **Timelapse MP4** de la nuit, avec étirement GHS (Generalized Hyperbolic
  Stretch), correction des couleurs et soustraction du fond de ciel
- **Filés d'étoiles** (star trails)
- **Détection et filtrage des satellites** (transformée de Hough)
- **Empilement avec alignement** sur les étoiles
- **Aperçu** du fond de ciel et du filtre satellites avant traitement

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`astropy` n'est nécessaire que pour les fichiers FITS.

## Lancement

```bash
./run_allsky.sh
```

Les réglages sont enregistrés dans `~/.config/allsky_app/settings.json`.
