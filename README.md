# CIFAR10Classification

This is for CIFAR10


Install Anaconda from the following link:
https://www.anaconda.com/products/individual

You need to scroll down to the end of the page to reach the download section.
Find the link 64-Bit Graphical Installer for python 3.7 according to your link.
Download and install Anaconda.

Then install Git using the following link:
https://git-scm.com/downloads

From your OS task bar Open Anaconda prompt(Anaconda3).
Type the following commands in conda.
conda create -n env571 python=3.6 ipykernel
conda activate env571


git clone https://github.com/HamidTohidypour/CIFAR10Classification

git pull

cd CIFAR10Classification

pip install -r requirements.txt

Add kernel:
python -m ipykernel install --user --name env571

jupyter notebook

Open the ipynb file