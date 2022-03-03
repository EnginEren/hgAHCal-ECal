#!/bin/bash


source /home/init_ilcsoft.sh
source /spack/share/spack/setup-env.sh

export REC_MODEL=ILD_l5_o1_v02
export MAIN=$(echo $PWD)

git clone --branch v02-02 https://github.com/iLCSoft/ILDConfig.git
cd ./ILDConfig/StandardConfig/production


export EOS_home=/eos/user/e/eneren
export LCIO=$(cat $1)
r=$3


echo "-- Running Reconstruction--"

Marlin MarlinStdReco.xml --constant.lcgeo_DIR=$lcgeo_DIR \
        --constant.DetectorModel=${REC_MODEL} \
        --constant.OutputBaseName=$EOS_home/run_$r/pion-shower_$2 \
        --constant.RunBeamCalReco=false \
        --global.LCIOInputFiles=$LCIO


echo $EOS_home/run_$r/pion-shower_$n\_REC.slcio > /mnt/lcio_rec_path
