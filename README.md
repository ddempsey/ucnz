# University of Canterbury teaching notebooks
A set of Jupyter Notebooks for teaching well testing, hydrology, reservoir engineering and numerical methods.

## Notebook index:

1. Well tests [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/ucnz/HEAD?filepath=well_test.ipynb)

2. Well siting [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/ucnz/HEAD?filepath=well_siting.ipynb)

3. Hydrology [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/ucnz/HEAD?filepath=hydrology.ipynb)

4. Wairakei Lumped Parameter Model [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/ucnz/HEAD?filepath=wairakei.ipynb)

5. ENCN304 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/ucnz/HEAD?filepath=ENCN304.ipynb)

6. Model calibration [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/ucnz/HEAD?filepath=calibration.ipynb)

7. Model uncertainty [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ddempsey/ucnz/HEAD?filepath=uncertainty.ipynb)

## Local Installation

Ensure you have Anaconda Python 3.X installed. Then

1. Clone the repo

```bash
git clone https://github.com/ddempsey/ucnz
```

2. CD into the repo and create a conda environment

```bash
cd ucnz

conda env create -f environment.yml

conda activate ucnz
```

3. Add the conda environment so it is accessible from the Jupyter Notebook

```bash
python -m ipykernel install --user --name=ucnz
```

## Use

If you are a local user, open a Jupyter Notebook server from the terminal

```bash
jupyter notebook
```

In the local server, or via the binder link, open a notebook. In the local server, select the `ucnz` environment in `Kernel > Change kernel`.

Run the notebook cells.

A document has been included in the repository with questions to test your understanding of the pumping test concepts.

## Author

David Dempsey (Department of Civil and Natural Resource Engineering, University of Canterbury)
