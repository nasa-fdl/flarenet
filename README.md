# solar-forecast
Space Weather Forecasting

## Folders ##

* training: the code for generating trained neural networks
* models: the trained output from the training folders
* data: data used for rapid prototyping and local evaluation. The full datasets should be at `/datasets` on the server

## Setting Up Your Local Environment ##

You are responsible for managing your own version of Python on the servers. These are the commads that are generally required for setting up the environment:

> wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh; chmod a+x Anaconda2-4.4.0-Linux-x86_64.sh; ./Anaconda2-4.4.0-Linux-x86_64.sh

> conda update --all  
> conda install astropy pydot graphviz keras  
> pip install tensorflow tensorflow-gpu  

Please update this based on your own experience.
