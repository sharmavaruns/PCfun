"""Install package."""
import os
from setuptools import setup, find_packages

REQUIREMENTS = []
if os.path.exists('requirements.txt'):
    for line in open('requirements.txt'):
        REQUIREMENTS.append(line.strip())

DESCRIPTION = (
    'PCfun, a hybrid supervised and unsupervised machine learning/text mining '
    'framework for the functional annotation of protein complex queries'
)

if os.path.exists('README.md'):
    LONG_DESCRIPTION = open('README.md').read()
else:
    LONG_DESCRIPTION = DESCRIPTION

scripts = ['bin/pcfun']

setup(
    name='pcfun',
    version='0.0.1',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='Varun Sharma, Matteo Manica, Chen Li',
    author_email=(
        'varunsharma.us@gmail.com, tte@zurich.ibm.com, chen.li@monash.edu'
    ),
    packages=find_packages('.'),
    install_requires=REQUIREMENTS,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    scripts=scripts
)
