import os

def change_directory_to_root():
    """
    Ensure the script is executing from the root of the project.
    """
    abspath = os.path.abspath(__file__)
    head, tail = os.path.split(abspath)
    os.chdir(head + "/..")
