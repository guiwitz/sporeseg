from setuptools import setup

setup(name='spores',
      version='0.1',
      description='Automated analysis of microscopy images of spores',
      url='https://github.com/guiwitz/sporeseg',
      author='Guillaume Witz',
      author_email='',
      license='BSD3',
      packages=['spores'],
      zip_safe=False,
      install_requires=['numpy','scikit-image','scikit-learn','jupyter','jupyterlab','pandas','tifffile','tqdm', 'matplotlib','requests'],
      )