#!/bin/bash


source /home/init_ilcsoft.sh
source /spack/share/spack/setup-env.sh

export REC_MODEL=ILD_l5_o1_v02
export MAIN=$(echo $PWD)

git clone --branch v02-02 https://github.com/iLCSoft/ILDConfig.git
cd ./ILDConfig/StandardConfig/production



#run=$(echo $1 | cut -d'/' -f3 )

echo "-- Running Reconstruction--"

Marlin MarlinStdReco.xml --constant.lcgeo_DIR=$lcgeo_DIR \
        --constant.DetectorModel=${REC_MODEL} \
        --constant.OutputBaseName=pion_shower-$1 \
        --constant.RunBeamCalReco=false \
        --global.LCIOInputFiles=$1


echo $1
#mv pion_shower-$run\_REC.slcio /mnt/$run
#ls -ltrh /mnt/$run