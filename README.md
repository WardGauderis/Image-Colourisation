# Image Colorization

Image Colorization project for the course Deep Learning at the VUB (2022-2023)

Fabian Denoodt, Ward Gauderis, Pierre Vanvolsem

---

The project is made in the form of a Jupiter Notebook. The notebook can be run on Google Colab, or locally
after installing the required packages from `requirements.txt` with the command `pip install -r requirements.txt`.

```bash
jupyter notebook colorization.ipynb
```

This notebook was used to train multiple models, test them and visualize the results.
But the actual model code is organised in classes in the accompanying Python files.

**Overview of the directory structure:**

- `colorization.ipynb`: The Jupiter Notebook
- `dataset.py`: The Dataset class
- `model.py`: The Model class
- `euclidean_model.py`: The Euclidean Model class with a different loss function
- `simplified_model.py`: The Simplified Model class with fewer parameters
- `requirements.txt`: The required packages to run the notebook
- `CHECKPOINTS`: The directory containing the final checkpoints of the trained models. In total 9 models were trained
  we try to include the most important models depending on the allowed upload file size.
- `DATA: The directory containing the data used for experimentation. The dataset is not included but is
  downloaded automatically when running the notebook.
- `DATA_FINAL`: The directory containing the data used for training the final models. The dataset is not included but is
  downloaded automatically when running the notebook.
- `DOWNLOADER`: The directory containing the scripts used to download the data. These are also downloaded automatically
  when running the notebook.
- `GRAPHS`: The directory containing all made visualizations.
- `PRECALCULATED`: The directory containing all precalculated data: the mapping between Q and AB values and multiple
distributions over the Q values.

# train patches to avoid overfitting

# report: sky and inside alwyas

# grass/ocean mistaken sometimes

# Cars multiple colors: harder, SSIM/PSNR can't acknowledge multiple are plausible

# First learn grey then color

# Simplified learns faster

# weights not smoothed -> better saturated extreme colors

# final worse -> more classes, do actually better on original test set

# Busier images more difficult not in paper

# temperature less effect in our model because weights are sharper, smoothends borders between colors