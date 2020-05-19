# CIFAR10Classification

This is a simple program classification of CIFAR10


# Install Anaconda 
from the following link:
https://www.anaconda.com/products/individual

You need to scroll down to the end of the page to reach the download section.
Find the link 64-Bit Graphical Installer for python 3.7.
Download and install Anaconda.

Then install Git using the following link:
https://git-scm.com/downloads

# Create a Virtual enviroment
From your OS task bar open Anaconda prompt(Anaconda3).
Type the following commands in conda:

conda create -n env571 python=3.6 ipykernel

conda activate env571


git clone https://github.com/HamidTohidypour/CIFAR10Classification

git pull

cd CIFAR10Classification

pip install -r requirements.txt

#Add kernel:
python -m ipykernel install --user --name env571

#Open the Jupyter:
jupyter notebook

Open the CNNCIFAR10.ipynb


# Anaconda for Mac
In order to install Anaconda on Mac:
Install pyenv (if you don't have it)
$ brew update # get brew if you don't have it https://brew.sh

$ brew upgrade

$ brew install pyenv

# check installation
$ pyenv --version

Install Anaconda (if you don't have it)
#Please check available versions of anaconda
$ pyenv install -l | grep anaconda
#install a specific version. (anaconda3-2019.10 or anaconda3-5.3.0 or anaconda3-5.3.1 was latest today 19th May 2020 ) It takes a bit of time cause it’s huge. This time it took me around 7min.
$ pyenv install anaconda3-2019.10
#To activate you need to 
$ pyenv global anaconda3-2019.10
#please replace with your python version
#Add path object by 
$ export PATH=$PATH: ~/user/nusrat/pyenv/anaconda3-2019.10/ # please adjust your path

$ conda create --name py python=3.7 anaconda 

And check if you can run commands installed in this process.

$ anaconda 

$ conda 

$ jupyter