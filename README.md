# Deep Learning for Solar Modeling

This repository defines an experimental environment for solar deep learning research. The initial problem solved by the repository is x-ray flux prediction, i.e. solar flare prediction. However, many solar modeling problems, including forecasting and understanding solar dynamics, require the same datasets. Our intention is to publish this repository so heliophysicists can build on its foundation to improve the x-ray flux forecasts and explore fundamental questions of heliophysics. We encourage anyone developing on top of this code base to open [pull requests](https://help.github.com/articles/about-pull-requests/) to improve our collective understanding of deep learning heliophysics research. Please note that neural network models can be combined to collectively predict or understand solar phenomena better than our piecewise efforts.

**Citing:** We ask that anyone using this deep learning framework to cite us as the following, TODO (bibtek, etc)

## Usage ##

This code base serves two purposes simultaneously. First, it has several models for forecasting solar events like solar flares. Second, it provides a set of analysis tools for deep learning models to inform the development of heliophysics research.

### Forecasting ###

The forecast models trained within this repository are fore scientific purposes only. Additional efforts to validate the results for operational settings are required.

### Science ###

Neural networks find high-dimensional interactions that may never be apparent to a person manually inspecting the data. There are several tools for extracting interpretable insights from these high dimensional interactions. We incorporate three of these tools into the repository, including,

**Saliency Maps:** [Saliency maps](https://arxiv.org/pdf/1312.6034.pdf) show the region trained neural networks pay attention to (i.e., are the most "salient") as a method for understanding why the network is making a particular forecast. For example,

> python science/saliency/tsne.py network_models/xray_flux_forecast/fdl1/trained_models/201707261524/best_performing.hdf5 dataset_models/SDO/AIA/fdl_1.py /data/sw/version1/x/20140111_0800_AIA_08_1024_1024.dat

will generate an image highlighting the regions of the sun that are most determinative of the forecast, i.e.

todo


**Embeddings:** Generating [embeddings](https://www.tensorflow.org/get_started/embedding_viz) uncovers sun states that the neural networks see as similar. For example, you can look at several images of the sun that the network sees as similar to extract insights regarding what is actually determinative of solar state change. Example:

> python science/embedding/tsne.py network_models/xray_flux_forecast/fdl1/trained_models/201707261524/best_performing.hdf5 dataset_models/SDO/AIA/fdl_1.py

will generate an embedding that you can explore within the visualization platform TensorBoard, i.e.

todo

**Deconvolution:** Generating [deconvolutional visualizations](https://arxiv.org/abs/1311.2901) uncovers the structured objects that are important for the network. For example, deconvolving on an image of the current x-ray flux output may show that regions with paired increases in flux output are predictive of changes in the flux output. Example:

> python science/deconvolutional/deconvolutional.py network_models/xray_flux_forecast/fdl1/trained_models/201707261524/best_performing.hdf5 dataset_models/SDO/AIA/fdl_1.py /data/sw/version1/x/20140111_0800_AIA_08_1024_1024.dat

will deconvolve the network to show the following image:

todo

## Technologies ##

todo: explain Python, Keras, Tensorflow,

For more details on these technologies, we recommend starting with the glossary before reading TODO.

## File Structure ##

    # Scripts for training models
    network_models/
        training_callbacks.py # Library for writing outputs during training
        xray_flux_forecast/ # Problem title
            README.md # File describing xray flux forecast problem in general
            fdl1/
                aia_only.py # Python script for training xray flux network on AIA data only
                # Trained neural networks output during training
                trained_models/
                    TRAINING_START_TIME/
			best_performing.hdf5 # The best performing network
			summary.txt # The summary output of this architecture
			epochs/ # models for each epoch
                        performance/ # performance for the architecture
                        maps/ # Outputs from generating a saliency map from the network
                        features/ # Outputs from deconvolving the network
                        embeddings/ # Outputs from generating network embeddings
        smac/ # Directory for network optimization library
            optimize.sh # The script for optimizing network architectures

    # Scripts for data access stored on disk
    dataset_models:
        solar_data.py # The abstract class for loading batches of data
        SDO/
            AIA/
                fdl_1.py # 8 Channel AIA data
            HMI/

    # Useful functions that are shared among many different scripts
    tools/
        tools.py

    # The software development tests for ensuring the networks are doing what they should do
    software_tests:
        tests.py  # the test suite

    # Python scripts for interpreting the data
    science:
        saliency/
            saliency.py # open a network and an image and output a map of the saliency
            maps/
        deconvolutional/
            deconvolutional.py # generate a deconv feature map
            features/
        embedding/
            tsne.py # generate a t-sne embedding for the data
            embeddings/

## Datasets ##

todo: instructions or scripts for downloading the required datasets. Preferably a script that will download then validate the integrity of the data.

## Hardware Requirements ##

Most of these experiments are only efficiently computable with a good GPU. Our hardware details are below, but you could likely run these experiments with less impressive infrastructure. We cannot be certain which elements of our compute stack are critical beyond having a GPU. You can also run these experiments with a CPU, but it will take several orders of magnitude longer to complete. During our initial phase of development we were provided with 16 of these machines by IBM.

<details> 
  <summary>Hardware </summary>
    <pre>
   Architecture:          x86_64
   CPU op-mode(s):        32-bit, 64-bit
   Byte Order:            Little Endian
   CPU(s):                56
   On-line CPU(s) list:   0-55
   Thread(s) per core:    2
   Core(s) per socket:    14
   Socket(s):             2
   NUMA node(s):          2
   Vendor ID:             GenuineIntel
   CPU family:            6
   Model:                 79
   Model name:            Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
   Stepping:              1
   CPU MHz:               1668.367
   CPU max MHz:           3500.0000
   CPU min MHz:           1200.0000
   BogoMIPS:              5201.29
   Virtualization:        VT-x
   L1d cache:             32K
   L1i cache:             32K
   L2 cache:              256K
   L3 cache:              35840K
   NUMA node0 CPU(s):     0-13,28-41
   NUMA node1 CPU(s):     14-27,42-55
   Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb intel_pt tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm rdseed adx smap xsaveopt cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm arat pln pts 
  </pre>
</details>

<details> 
  <summary>GPUs </summary>
    <pre>
        Wed Jul 26 16:18:35 2017       
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |===============================+======================+======================|
        |   0  Tesla P100-PCIE...  Off  | 0000:81:00.0     Off |                    0 |
        | N/A   41C    P0    27W / 250W |      0MiB / 16276MiB |      0%      Default |
        +-------------------------------+----------------------+----------------------+
        |   1  Tesla P100-PCIE...  Off  | 0000:82:00.0     Off |                    0 |
        | N/A   42C    P0    28W / 250W |      0MiB / 16276MiB |      3%      Default |
        +-------------------------------+----------------------+----------------------+
                                                                               
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID  Type  Process name                               Usage      |
        |=============================================================================|
        |  No running processes found                                                 |
        +-----------------------------------------------------------------------------+
    </pre>
</details>

## Setting Up Your Modeling Environment ##

Requirements:

* nvidia's compiler and CUDA libraries
* Anaconda Python installation

Example setup script for a server that already has the core Nvidia software installed.

> wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh; chmod a+x Anaconda2-4.4.0-Linux-x86_64.sh; ./Anaconda2-4.4.0-Linux-x86_64.sh

> conda update --all  
> conda install astropy pydot graphviz keras  
> pip install tensorflow tensorflow-gpu  

Once you successully execute this installation, you should then clone this repository and add it to your python path.

> git clone REPOSITORY_REFERENCE_HERE
> cd solar-forecast
> export PYTHONPATH=/PATH/TO/REPOSITORY/solar-forecast:$PYTHONPATH

You can add the line about exporting to your .bash_profile or .bashrc file so you will not need to type it in on every connection to your server.

Please open pull requests to correct this setup script based on your own experiences.

todo: provide guide on SMAC installation

## Deep Learning Glossary ##

todo
