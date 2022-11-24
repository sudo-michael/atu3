# atu3

## Install Anaconda
Install the anaconda from the [website](https://docs.anaconda.com/anaconda/install/linux/)

## Clone the Repositories from GitHub

```
git clone https://github.com/sudo-michael/atu3.git
```
```
git clone https://github.com/SFU-MARS/optimized_dp.git
```

## Navigate to the *atu3* Folder
```
cd PATH/TO/FOLDER 
```

## Create new Everoment from conda
```
conda env create -f environment.yml
```

```
conda activate atu3
```

```
conda install -c cornell-zhang heterocl -c conda-forge
```

```
pip install stable-baselines3
```

```
pip install gym==0.23.1
```

from outsite the folder, install the modules
```
pip install -e atu3
pip install -e optimized_dp
```
