# CIFAR10Classification

This is a simple program for classification of CIFAR10 with 10 classes


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

#Setup the virtual environment:

  conda create -n env571 python=3.6 ipykernel

#Activate the virtual environment:

  conda activate env571

# Download the code and install the dependencies:
git clone https://github.com/HamidTohidypour/CIFAR10Classification

  git pull

  cd CIFAR10Classification

#install the dependencies: 

  pip install -r requirements.txt


#Add kernel for env571:

  python -m ipykernel install --user --name env571






# Two ways of running the code

#1-Running the code from the Anaconda's command line:

  python CNNCIFAR10.py

#2-Using Jupyter notebook:

  jupyter notebook



#Open the CNNCIFAR10.ipynb in jupyter


# Anaconda for Mac
In order to install Anaconda on Mac:
Install pyenv (if you don't have it)
$ brew update # get brew if you don't have it https://brew.sh

  brew upgrade

  brew install pyenv

# Check installation (for Mac)
  pyenv --version

Install Anaconda (if you don't have it)
#Please check available versions of anaconda
  pyenv install -l | grep anaconda

#install a specific version. (anaconda3-2019.10 or anaconda3-5.3.0 or anaconda3-5.3.1 was latest today 19th May 2020 ) It takes a bit of time cause it’s huge. This time it took me around 7min.

  pyenv install anaconda3-2019.10
#To activate you need to 

  pyenv global anaconda3-2019.10
#please replace with your python version
#Add path object by 
  export PATH=$PATH: ~/user/yourusername/pyenv/anaconda3-2019.10/ # please adjust your path

  conda create --name py python=3.7 anaconda 

And check if you can run commands installed in this process.

  anaconda 

  conda 

  jupyter