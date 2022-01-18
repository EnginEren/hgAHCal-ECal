#!/bin/bash

source /home/init_ilcsoft.sh
source /spack/share/spack/setup-env.sh

export SIM_MODEL=ILD_l5_v02
export MAIN=$(echo $PWD)

git clone --branch v02-02 https://github.com/iLCSoft/ILDConfig.git
cd ILDConfig/StandardConfig/production
git clone https://github.com/iLCSoft/lcgeo.git

cp $MAIN/geo/*.xml ./lcgeo/ILD/compact/ILD_common_v02/
cp $MAIN/pionGun.mac . 
cp $MAIN/ddsim_steer_macro.py .
cp $MAIN/create_root_tree.xml .

n=$1
r=$2

echo "-- Running DDSim..."
ddsim --outputFile ./pion-shower_$n.slcio --compactFile ./lcgeo/ILD/compact/${SIM_MODEL}/${SIM_MODEL}.xml --steeringFile ddsim_steer_macro.py 

echo "Converting: LCIO --> root file"
Marlin create_root_tree.xml --global.LCIOInputFiles=./pion-shower_$n.slcio --MyAIDAProcessor.FileName=pion-shower_$n;

mkdir /mnt/run_$r && mv ./pion-shower_$n.slcio /mnt/run_$r && mv ./pion-shower_$n.root /mnt/run_$r

echo /mnt/run_$r/pion-shower_$n.slcio > /mnt/lcio_path
echo /mnt/run_$r/pion-shower_$n.root > /mnt/root_path