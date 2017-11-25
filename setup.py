from setuptools import setup
from setuptools import find_packages


setup(name='FlareNet',
      version='0.2.0',
      description='Deep Learning Library for Solar Modeling',
      author='Frontier Development Lab',
      author_email='FlareNet@seanbmcgregor.com',
      url='https://github.com/nasa-fdl/solar-forecast',
      download_url='https://github.com/nasa-fdl/solar-forecast/',
      license='AGPL',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml'],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot>=1.2.0'],
          'tests': ['pytest'],
      },
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: AGPL License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
