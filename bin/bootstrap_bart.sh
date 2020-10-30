#!/bin/bash

conda create -n zest_bart python=3.7
conda activate zest_bart
pip install -r requirements.txt
pip install -r dev-requirements.txt
pip install -r bart-requirements.txt

# get transformers
echo "INSTALLING transformers"
git clone git@github.com:huggingface/transformers.git
# This is the 3.3.1 branch
cd transformers
echo `pwd`
git checkout 1ba08dc221ff101a751c16462c3a256d726e7c85
echo "Installing 3.3.1 branch"
git status

pip install -e .
pip install -r ./examples/requirements.txt

# We link these directories and apply some namespace mangling so that it's possible
# to import and subclass some modules in transformers/examples.
cd ..
ln -s transformers/examples transformers_examples

python setup.py install

