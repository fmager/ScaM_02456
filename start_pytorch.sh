#!/bin/bash -e
module load LUMI/24.03 partition/G PrgEnv-amd PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315
wd=/users/magerfab/PhD/ijepa
singularity exec -B $wd:/workdir $SIF /bin/bash  -c "export PYTHONPATH=\"${PYTHONPATH}:/users/magerfab/PhD/ScaM\"; exec /bin/bash"
