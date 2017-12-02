# Deep Learning for Solar Modeling

FlareNet defines an experimental environment for deep learning research with images of the sun. The initial problem introduced by the repository is x-ray flux prediction, i.e. solar flare prediction. However, the framework is appropriate for all solar modeling problems where the independent variables are solar images. Our purpose in publishing FlareNet is to facilitate collaboration between heliophysicists and deep learning researchers. We encourage anyone developing on top of this code base to open [pull requests](https://help.github.com/articles/about-pull-requests/) to advance our collective efforts at understanding of solar physics.

**Citing:** We ask that anyone using this deep learning framework to cite FlareNet as the following:

    McGregor, S., Dhuri, D., Berea, A., & Munoz-Jaramillo, A. (2017). FlareNet: A Deep Learning Framework for Solar Phenomena Prediction. In NIPS Workshop on Deep Learning for Physical Sciences. Long Beach.

    @inproceedings{McGregor2017,
        address = {Long Beach},
        author = {McGregor, Sean and Dhuri, Dattaraj and Berea, Anamaria and Munoz-Jaramillo, Andres},
        booktitle = {NIPS Workshop on Deep Learning for Physical Sciences},
        title = {{FlareNet: A Deep Learning Framework for Solar Phenomena Prediction}},
        year = {2017}
    }


## Usage ##

The core commands pertain to training, testing, and visualization. Each of these commands have a common set of parameters.

After completing installation, you run experiments by copying one of the network model folders and issuing training runs, with, for example, `python network_models/xray_flux_forecast/starting_point/network.py`.

## Installation ##

You can run FlareNet either on your local machine or on [Amazon Web Services](https://aws.amazon.com/amazon-ai/amis/) (AWS). In both instances you will need a large amount of disk storage (around a terabyte) for the files that will be downloaded.

### Local Setup ###

To setup FlareNet on your local machine, you should install Anaconda's Python. You should additionally install NVIDIA's compiler and CUDA libraries if you are training on a GPU.

Example setup script for a server that already has the core Nvidia software installed.

> wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh; chmod a+x Anaconda2-4.4.0-Linux-x86_64.sh; ./Anaconda2-4.4.0-Linux-x86_64.sh

> conda update --all  
> conda install astropy pydot graphviz keras  
> pip install tensorflow tensorflow-gpu feather-format  

Once you successfully install, you should then clone this repository and add it to your python path.

> git clone REPOSITORY_REFERENCE_HERE
> cd flarenet
> export PYTHONPATH=/PATH/TO/REPOSITORY/flarenet:$PYTHONPATH

You can add the line about exporting to your .bash_profile or .bashrc file so you will not need to type it in on every connection to your server.

Please open pull requests to correct this setup script based on your own experiences.

### AWS Setup ###

Amazon Web Services (AWS) provides cloud computing machines with powerful GPUs for training your models. We are working with the [AMI Deep Learning Image](https://aws.amazon.com/marketplace/pp/B076T8RSXY?qid=1509927949185) for Amazon Linux that pre-packages 
CUDA 9, cuDNN 7, NCCL 2.0.5, NVidia Driver 384.81, and TensorFlow [master with enhancements for CUDA 9 support].

**Selecting your machine:** If you are on a budget, we recommend working with a p2.xlarge machine. These are equipped with K80 GPUs, which have more GPU memory than a single p3 machine. The p3 machines share GPU memory which makes them better when you use one of the more expensive machines. Either way, the memory available is the most important factor determining training efficiency for these solar problems. The images are large, multi-channeled, and noisy. The bigger the batch size, the better. Also note, you can use the cheapest instance when downloading data to Amazon storage. You can switch to a bigger instance for training after you have downloaded all the data you want.

Steps:

1. [Create an account with AWS](https://portal.aws.amazon.com/billing/signup?nc2=h_ct&redirect_url=https%3A%2F%2Faws.amazon.com%2Fregistration-confirmation#/start)
2. [Create a machine instance](https://us-west-2.console.aws.amazon.com/ec2/v2/home?region=us-west-2#LaunchInstanceWizard:) Select the "Deep Learning Base AMI (Amazon Linux) Version 1.0 - ami-dceb3aa4" instance from the list.
3. Select the hardware you want to deploy to, then launch the instance. See "Selecting your machine" above for details.
4. Connect to the newly launched instance over SSH. This should be something like `ssh ec2-user@YOUR-MACHINE-HERE.us-west-2.compute.amazonaws.com`
5. Activate the software required by FlareNet `source activate tensorflow_p27`
6. Clone the FlareNet repository, `git clone https://github.com/nasa-fdl/flarenet.git`
7. Change into the cloned directory `cd flarenet`
8. Add the flarenet directory to the Python search path. `export PYTHONPATH=/home/ec2-user/flarenet:$PYTHONPATH`
9. Install the required libraries with the following 3 commands `conda update --all`, `conda install astropy pydot graphviz`, `pip install feather-format`
9. Now we are going to create the disk that will store all the SDO images we download. Since we may eventually want to migrate the data from Elastic Block Storage (EBS; which is faster for an individual machine but doesn't scale to multiple machines) to Elastic File System (EFS; which scales to a cluster of GPU machines), we are going to create a separate EBS volume. The separate EBS volume will make it easier to prototype on cheap hardware, then scale to progressively more expensive GPU instances. When the largest instance offered by Amazon does not suffice, we will want to migrate to EFS (costs 3x more) and begin cluster computing on multiple machines. For now, we recommend going to the [EBS volume creation page](https://us-west-2.console.aws.amazon.com/ec2/v2/home?region=us-west-2#CreateVolume:) and creating a separate volume. Start out with a small volume size and dial it up later as you download more SDO images.
10. [Attach the new EBS volume](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html) to the running machine by clicking on the EBS volume then selecting the "attach" action. Change the "Device" setting from `/dev/sdf` to `/dev/sdo`.
11. Now we are going to format the device volume as attached to our machine with the command `sudo mkfs -t ext4 /dev/xvdo`
12. Now create a mount point for the drive with `sudo mkdir /data`.
13. Mount the drive to the mount point with `sudo mount /dev/xvdo /data`
14. Now you have the machine to run experiments and the place to store data. It is time to issue a test run: `python network_models/synthetic/version0/network.py` This should start training on a synthetic solar activity generator.
15. Now we are going to start working with the SDO data, but first we need to download it from servers hosted at Stanford University. This download operation can take a long time, so we recommend starting the operation in a [screen](https://www.gnu.org/software/screen/manual/screen.html#Overview) session. If you don't want to figure out screen you can just have the terminal window sit open for a few days. Issue the following command to begin downloading all the flaring instances: `python dataset_models/sdo/download/download_aia.py`.
16. Now let's separate the training and the testing sets with the following command, `python dataset_models/sdo/structure_data.py`. If you do not download the whole dataset this will default to separating 100 flaring cases into the validation set, or 10 percent of all the data (whichever is smaller).
17. Now it is time to generate the dependent variable file based on the data you downloaded. This will be pulled from the GOES satellites, which measure solar xray flux output. Issue `python tools/xray_flux_forecast/v4_find_ys.py` and the dependent variable files will be created.
18. Now you can do a test run of training on the full data with `python network_models/xray_flux_forecast/starting_point/network.py`.

## File Structure ##

Assuming you are working on the x-ray flux prediction problem, you should generally start by copying the "starting_point" project to a new directory. It includes a rudimentary network specification from which you can customize. Issue this command from the root of the repository: `cp network_models/xray_flux_forecast/starting_point network_models/xray_flux_forecast/my_network_experiment`

    # Scripts for training models.
    network_models/
        training_callbacks.py # Library for writing outputs during training
        experiment.py # The file that structures the experiments you run. You should not need to change this
        xray_flux_forecast/ # All networks in this folder are for solar flare prediction
            starting_point/
                config.py # The network specific configuration
                network.py # The neural network specification file
                # Trained neural networks output during training
                trained_models/
                    TRAINING_START_TIME/
                        summary.txt # The summary output of this architecture
                        epochs/ # models for each epoch
                        performance/ # performance for the architecture
                        maps/ # Outputs from generating a saliency map from the network
                        features/ # Outputs from deconvolving the network
                        embeddings/ # Outputs from generating network embeddings

    # Scripts for data access stored on disk
    dataset_models:
        dataset.py # The abstract class for loading batches of data
        sdo/
            aia.py # 8 Channel AIA data
            layers.py # Custom neural network layers for processing data on the GPU

    # Useful functions that are shared among many different scripts
    tools/
        tools.py

    # The software development tests for ensuring the networks are doing what they should do
    software_tests:
        testing_installation.py  # Test for the presence of all the required libraries
        synthetic.py  # Test the synthetic sun domain

## Deep Learning Glossary ##

* [Keras](https://keras.io/): A library for specifying neural networks. It build on top of TensorFlow and other Deep Learning libraries with a cleaner way of specifying neural network architectures. If you want to know how to specify a new neural network architecture on FlareNet, then you should look at the Keras API.
* [Tensorflow](https://www.tensorflow.org/): Keras builds on top of TensorFlow and TensorFlow provides the efficient methods for training and evaluating neural networks on either GPUs or CPUs. You will need to know TensorFlow if there is functionality that you need to build that is not supported by Keras.
* [CUDA](https://en.wikipedia.org/wiki/CUDA): CUDA is the low level language for parallel computing developed by NVIDIA. You should not need to develop anything in this language since TensorFlow abstracts away these complexities.
* Is something not covered here? You can add additional terms by opening a pull request. Opening these simple pull requests is a great way to open the discussion.

## Hardware Requirements ##

Most of the models only train efficiently with a good GPU. FlareNet was originally developed for the hardware below, but you can run these experiments with less impressive infrastructure. You can also run these experiments with a CPU, but it will take several orders of magnitude longer to complete. During our initial phase of development we were provided with 16 of these machines by IBM.

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


## Credits ##

This project makes extensive use of Open Source technologies that deserve credit for their development.

We are also very grateful to IBM and IBM’s Troy Hernandez and Naeem Altaf for their incredible support in terms of hardware and tech. Without their support this work would not have been possible. We are very grateful to Kx, and to NASA’s Heliophysics division for providing financial support that enabled this interdisciplinary collaboration and to the SETI Institute for providing the logistic framework that nurtured and enabled our research.

We are grateful to Mark Cheung, Monica Bobra, Ryan McGranaghan, and Graham Mackintosh, co-mentors at NASA’s Frontier Development Laboratory, for valuable suggestions and discussions.

The X-ray Flare dataset was prepared by the NOAA National Geophysical Data Center (NGDC). SDO data were prepared and made available for download by Stanford’s Joint Science Operations Center (JSOC).

## Contact ##

To contact the developers of this software, we recommend opening an issue on this repository or emailing flarenet at the domain "seanbmcgregor.com".
