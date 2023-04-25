from setuptools import setup
import unittest


def readme():
    with open('README.rst') as f:
        return f.read()


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir='tests',
        pattern='test_*.py',
        top_level_dir='eeg_visualizer')
    return test_suite


setup(name='eeg_visualizer',
      version='0.0.1',
      description='Metrics for evaluating survival analysis models',
      long_description=readme(),
      url='http://github.com/Navid-Ziaei/EEG-Visualizer',
      author='Navid Ziaei',
      author_email='n2ziaee@gmail.com',
      license='MIT',
      packages=['eeg_visualizer'],
      install_requires=[
          'numpy', 'pandas', 'tqdm', 'matplotlib', 'scikit-learn', 'scipy', 'seaborn',
          'dash', 'joblib', 'colorcet', 'plotly', 'nitime'
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='setup.my_test_suite',
      )
