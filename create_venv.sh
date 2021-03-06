#!/usr/bin/env bash

VENVNAME=lang_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

cat requirements.txt | xargs -n 1 -L 1 pip install

deactivate
echo "build $VENVNAME"
