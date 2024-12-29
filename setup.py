
from setuptools import setup, find_packages

setup(
    name='easy_glm',
    version='0.1',
    packages=find_packages(),   
    install_requires=[],
    author='Serban Dragne',
    author_email='sadragne@gmail.com',
    description='A package to quickly create insurance pricing models using GLMs and export models as ratetables',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)