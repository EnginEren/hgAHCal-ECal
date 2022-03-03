#!/bin/bash


source /home/init_ilcsoft.sh
source /spack/share/spack/setup-env.sh

export REC_MODEL=ILD_l5_o1_v02
export MAIN=$(echo $PWD)

git clone --branch v02-02 https://github.com/iLCSoft/ILDConfig.git
cd ./ILDConfig/StandardConfig/production


export EOS_home=/eos/user/e/eneren
r=$3


echo "-- Running Reconstruction--"

Marlin MarlinStdReco.xml --constant.lcgeo_DIR=$lcgeo_DIR \
        --constant.DetectorModel=${REC_MODEL} \
        --constant.OutputBaseName=pion-shower_$2 \
        --constant.RunBeamCalReco=false \
        --global.LCIOInputFiles=$EOS_home/$1


mv pion-shower_$2_REC.slcio $EOS_home/run_$r
