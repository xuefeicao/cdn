from setuptools import setup

readme = open('README.md').read()

setup(
    name="CDN",
    version="0.0.1",
    description="Implementations of Causal Dynamic Network Analysis of fMRI",
    author="Xuefei Cao, Xi Luo, Bjorn Sandstede",
    author_email="xcstf01@gmail.com",
    packages=['CDN'],
    long_description=readme,
    install_requires=[
        "matplotlib>=1.5.3",
        "numpy>=1.11.1",
        "scipy>=0.19.0",
        "six>=1.10.0",
        "scikit-learn>=0.18.1"
    ],
    url='https://github.com/xuefeicao/CDN',
    include_package_data=True,
    )
