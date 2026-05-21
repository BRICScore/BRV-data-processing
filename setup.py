from setuptools import setup, find_packages

setup(
    name='brics_toolkit',
    version='0.1',
    description='Toolkit for BRICS environment',
    author='BRICS Development team',
    packages=find_packages(),  # same as name
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'seaborn', 'requests', 'pyopenssl', 'pydantic'], # external packages as dependencies
)