# !/usr/bin/env python3

from setuptools import setup, find_packages


long_description = 'PCfun is a hybrid supervised and unsupervised machine learning/text mining framework for the functional annotation of protein complex queries, created using a fastText word embedding built upon 1 million open access articles in PubMed Central.'

setup_args = dict(
    name='PCfun',
    version='0.0.1',
    packages=find_packages(),
    scripts=[
#        'collapse.py',
#        'differential.py',
#        'exceptions.py',
#        'generate_features.py',
#        'go_fdr.py',
#        'io_.py',
#        'hypothesis.py',
#        'mcl.py',
#        'init.py'
#        'map_to_database.py',
#        'mcl.py',
#        'merge.py',
#        'parse_GO.py',
#        'main.py',
#        'plots.py',
#        'predict.py',
#        'stats_.py',
#        'validate_input.py',
    ],
    # long_description=long_description,
    license='BSD',
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['Cython','fastText','scipy>=1.1', 'pandas', 'sklearn', 'networkX'],
    package_data={
        'PCfun': [##TBD
        #'go_term_class.txt', 'go-basic.obo', 'rf_equal.clf'],
    },
    # metadata to display on PyPI
    author='Varun Sharma',
    author_email='varunsharma.us@gmail.com',
    description='Text-mining/machine learning prediction tool for the functional annotation of protein complex queries.',
    keywords=['proteomics', 'machine-learning', 'text-mining','protein complexes'],
    url='https://github.com/sharmavaruns/PCfun/',
    project_urls={
        'Bug Tracker': 'https://github.com/sharmavaruns/PCfun',
        'Documentation': 'https://github.com/sharmavaruns/PCfun',
        'Source Code': 'https://github.com/fossatiA/PCProphet/',
    },
    platforms="Linux, Mac OS X, Windows",
    long_description=long_description,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Modified BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)
# TODO add compilation for Java GUI


def main():
    setup(**setup_args)


if __name__ == '__main__':
    main()
