# #!/bin/bash

conda activate unicats
export PATH=$PWD/utils:$PWD/unicats/bin:$PATH  # to add `unicats/bin` into PATH so that train.py and decode.py will directly be called
export PYTHONPATH=$PWD:$PYTHONPATH
