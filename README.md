# PCfun (Protein Complex Function)

Fast and accurate tool for the functional annotation of protein complex queries built upon hybrid unsupervised and supervised machine learning on PubMed Central full-text word embeddings.

## Installation
### For installation you need to install the PCfun package and then download the required data files (embedding and pretrained supervised models)

#### 1) After having installed Github Desktop or some flavor of Git so that you can use Git from terminal, ***clone*** the pcfun repo to your place of choice.
For example in a folder called "Github" (that you manually create or Github Desktop has created for you. Whichever you prefer!) do the following:
```
mkdir Example
cd Example
mkdir Github
cd Github
git clone https://github.com/sharmavaruns/PCfun.git
cd PCfun
```

#### 2) Recommended to use conda/miniconda for installing a new python3 environment
```
conda create --name PCfun python=3.7.7
``` 
#### 3) Activate created conda environment then pip install PCfun
- For pygraphviz to work (which is used for visualizing GO graphs), graphviz needs to be installed. Therefore you need to first install pygraphviz separately.
- I'm unsure if the installation of pygraphviz will work on Linux due to the '/usr/...', will need to be tested.
- Then install the PCfun package and the relevant dependencies should be installed
```
conda activate PCfun
pip install pygraphviz==1.5 --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
pip install . #pip install PCfun ## Unsure which works
```

#### 4) After installing the dependencies, download the required PCfun data files (word embedding + trained supervised Random Forest classifiers)
First ensure that your created conda environment where PCfun has been installed into is active then do: 
```
cd Example/Github/PCfun
time python ./download.py
```
***NOTE:*** This will take awhile. For me it took ~1 hour as the zipped file being downloaded from S3 is 9.8 Gb.
The "time" command is optional, but gives you some idea of how long everything takes.

***This step downloads "pcfun.zip" from my public S3 bucket directly into your home directory (e.g. ~) by default and unzips it there.***


## Example Usage- Project0
#### 1) First we'll create a new directory for your project where all the results and relevant files will be stored
```
cd Example
mkdir PCfun_Projects
cd PCfun_Projects
mkdir Project0
cd Project0
cp Example/Github/PCfun/Toy_Data_Input/input_df-UniProtIDs.tsv .
```
#### 2) Now we'll activate our PCfun conda env and run pcfun on the input data file
```
conda activate PCfun
time pcfun -u -i input_df-UniProtIDs.tsv
```
***NOTE:*** This is now running PCfun on the input data set ("input_df-UniProtIDs.tsv") you've defined.
 The "time" command is optional, but gives you some idea of how long everything takes. Though I have some times reported in the pcfun script anyway.

***NOTE 2:*** 'pcfun -u -i input_df-UniProtIDs.tsv' is an example where the input expects a .tsv file with UniProt Subunit IDs delimited with ';' (hence the '-u' flag).
 See below for other example usage cases (input df consisting of: ComplexNames, UniProtIDs, or GeneNames).

***NOTE 3:*** This took ***~45 minutes for 50 example complexes*** to run on my 2020 MacBook Pro with a Processor: 2 GHz Quad-Core Intel Core i5 & Memory: 16 GB 3733 MHz LPDDR4X

### Note: I have included three possible use cases when calling the pcfun script based on the input data set
### The corresponding data sets used below are in 'PCfun/Toy_Data_Input' directory within cloned PCfun repo for ease of access.
#### - If input data set includes UniProt IDs for each subunit protein delimited by a ';' for a protein complex use the "-u" flag.
```
pcfun -u -i input_df-UniProtIDs.tsv
```

#### - If input data set includes the Gene Names (that have already been mapped from UniProt IDs) for each subunit protein delimited by a ';' for a protein complex use only the "-g" flag.
```
pcfun -g -i input_df-GeneNames.tsv
```

#### - If input data set includes the entire protein complex name (i.e. no subunits delimited with ';') then use no flag.
```
pcfun -i input_df-FullComplexNames.tsv
```
