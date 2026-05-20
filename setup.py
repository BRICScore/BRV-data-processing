from setuptools import setup

setup(
    name='brics_toolkit',
    version='0.1',
    description='Toolkit for BRICS environment',
    author='BRICS Development team',
    packages=['brics_types', 'data_containers', 'data_processing', 'database_access', 'utils'],  # same as name
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'seaborn', 'requests', 'pyopenssl', 'pydantic'], # external packages as dependencies
)