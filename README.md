# PCfun (Protein Complex Function)

Fast and accurate tool for the functional annotation of protein complex queries built upon hybrid unsupervised and supervised machine learning on PubMed Central full-text word embeddings.

## Note: Currently PCfun has been tested on Linux and Mac. I'll be working to test it for Windows soon (and perhaps create Docker version if dependencies are incompatible with Windows for some reason.)

## Installation
### For installation you need to install the PCfun package and then download the required data files (embedding and pretrained supervised models)

#### 1) After having installed Github Desktop or some flavor of Git so that you can use Git from terminal, ***clone*** the pcfun repo to your place of choice.
For example in a folder called "Github" (that you manually create or Github Desktop has created for you. Whichever you prefer!) do the following:
```
mkdir Example # This is an example dummy folder. Just for reference I have a Github folder in my Documents folder on Mac
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
- Then install the PCfun package and the relevant dependencies should be installed
```
conda activate PCfun
pip install pygraphviz==1.5 --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
### Note: "--include-path=/usr/include/graphviz" allows graphviz to be installed with pygraphviz
### This should cause graphviz to be downloaded to "/usr/lib/graphviz/" by default
### If you have independently installed graphviz, please direct the install command for pygraphviz accordingly

### Now install PCfun and the rest of its dependencies with the following
### Note: ensure that you're current directory is in the "PCfun" directory that you've cloned
pip install .
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
cd Example # Just navigating back to the Example directory
mkdir PCfun_Projects
cd PCfun_Projects
mkdir Project0
cd Project0
cp Example/Github/PCfun/Toy_Data_Input/input_df-UniProtIDs.tsv . # Copying over example toy data into our local project
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

***NOTE 4:*** By default PCfun looks for downloaded and unzipped data folder in your home directory of your computer. Otherwise, you can use the "-r" flag to direct PCfun to the downloaded directory. (e.g. "-r /Users/varunsharma/pcfun")
 
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

## OUTPUT: PCfun will now automatically create the following in your Project directory:
- "query_vecs.tsv": the continuous word embedding vectors for your input queries
- "Results": Directory with subdirectories named after each query
    - In each subdirectory you will have subdirectories: "BP_GO", "CC_GO", and "MF_GO"
    - In each "*_GO" subdirectory you will have "funcenrich_list.tsv" and "KDTree_list.tsv"
    - "funcenrich_list.tsv" corresponds to the Supervised RF's results and indicates if any of the terms were functionally enriched with the nearest neighbors results
    - "KDTree_list.tsv" corresponds to the ranked nearest neighbor results for the query
    - Lastly, an additional subirectory called "Tree_diags" may be created within each "*_GO" directory if any terms were functionally enriched for
        - If more then 10 terms are functionally enriched for, then only the top 10 functionally enriched GO trees will be plotted


## I will be working to put up the code used for creating the embeddings, training the supervised models, and generating all of the figures for sake of reproducibility.
If you have any questions on particular details here, feel free to reach out to me (varunsharma.us@gmail.com) and I'll be happy to answer any questions! 