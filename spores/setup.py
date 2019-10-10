from setuptools import setup

setup(name='spores',
      version='0.1',
      description='Automated analysis of microscopy images of spores',
      url='https://github.com/guiwitz',
      author='Guillaume Witz',
      author_email='',
      license='MIT',
      packages=['spores'],
      zip_safe=False,
      install_requires=['numpy','scikit-image','scikit-learn','jupyter','jupyterlab','pandas','tifffile','tqdm', 'matplotlib','requests'],
      )
