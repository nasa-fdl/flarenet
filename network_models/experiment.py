"""
This script coordinates training and writing of all the files surrounding the experiment.
"""
def experiment(network_model, output_path, dataset_model=None, args=None, config={}):
    """Function that runs the training. Call this from the network specification file.

    Positional arguments:
        network_model -- The network as specified by the Keras Functional API
        output_path -- The path to where the results of the runs will be written

    Keyword arguments:
        dataset_model -- The dataset model that will generate data for the network
        args -- The command line arguments that you want written to the output directory as a record
        config -- The configuration dictionary that is written to a config.yml file
    -- 
    """

    # Performing periodic actions during training
    from keras.callbacks import ModelCheckpoint

    # Visualization Library for tracking training
    from keras.callbacks import TensorBoard

    # Utilities for this script
    import os
    import sys

    # Libraries packaged with this repository
    from network_models.training_callbacks import TrainingCallbacks
    from tools import tools

    if "force CPU" in config and config["force CPU"]:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set the paths
    tools.change_directory_to_root()
    training_callbacks = TrainingCallbacks(output_path, args)

    #####################################
    #    Printing Architecture Record   #
    #####################################

    steps_per_epoch = config["steps_per_epoch"]
    samples_per_step = config["samples_per_step"] # batch size
    epochs = config["epochs"]

    # Print the netwrok summary information
    network_model.summary()
    orig_stdout = sys.stdout
    f = open(output_path + training_callbacks.timestr + "/summary.txt", 'w')
    sys.stdout = f
    network_model.summary() # This does not return the summary string so we capture standard out
    sys.stdout = orig_stdout
    f.close()

    print("##################")
    print("Run identifier: " + str(training_callbacks.timestr))
    print("You can find the results from this run in a folder named " + str(training_callbacks.timestr))
    print("##################")

    # Do not allow a configuration with more than 150 million parameters
    if network_model.count_params() > 150000000:
        print("exiting since this network architecture will contain too many parameters")
        print("Result for SMAC: SUCCESS, 0, 0, 999999999, 0") #  todo: figure out the failure string within SMAC
        exit()

    #####################################
    #   Optimizing the Neural Network   #
    #####################################

    # Save intermediate outputs including the full model
    tensorboard_log_data_path = "/tmp/" + output_path
    tensorboard_callbacks = TensorBoard(log_dir=tensorboard_log_data_path)
    model_output_path = output_path + training_callbacks.timestr + "/epochs/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    model_checkpoint = ModelCheckpoint(model_output_path)

    history = network_model.fit_generator(dataset_model.get_training_generator(),
                                       steps_per_epoch,
                                       max_queue_size=10,
                                       epochs=epochs,
                                       validation_data=dataset_model.get_validation_generator(),
                                       callbacks=[tensorboard_callbacks, training_callbacks, model_checkpoint],
                                       validation_steps=dataset_model.get_validation_step_count(),
                                       workers=1,
    )

    # Loss on the training set
    print("printing loss history")
    print(history.history['loss'])

    # Loss on the validation set
    if 'val_loss' in history.history.keys():
        print("printing history of validation loss over all epochs:")
        print(history.history['val_loss'])

    # Print the performance of the network for the SMAC algorithm
    print("Result for SMAC: SUCCESS, 0, 0, %f, 0" % history.history['loss'][-1])