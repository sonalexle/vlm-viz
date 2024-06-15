#!/bin/bash

rm -r ~/.cache
ln -s /nlp-rcp-scratch/home/le/.cache ~/.cache
source /nlp-rcp-scratch/home/le/envs/general-env/bin/activate
python -m ipykernel install --name=general-env --user
# python -q -m spacy download en_core_web_sm
#pip install -q -U accelerate bitsandbytes git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/diffusers.git git+https://github.com/huggingface/trl.git

# python -q -m spacy download en_core_web_lg
# sudo pip install -q -U pycocoevalcap accelerate bitsandbytes git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/diffusers.git git+https://github.com/huggingface/trl.git
# sudo python -q -c "from pycocoevalcap.spice.spice import Spice; Spice()"
# sudo chmod 777 /usr/local/lib/python3.10/dist-packages/pycocoevalcap/spice
