#!/bin/bash

source /home/init_ilcsoft.sh
source /spack/share/spack/setup-env.sh

export SIM_MODEL=ILD_l5_v02
export MAIN=$(echo $PWD)

git clone --branch v02-02 https://github.com/iLCSoft/ILDConfig.git
cd ILDConfig/StandardConfig/production


cp $MAIN/pionGun.mac . 
cp $MAIN/ddsim_steer_macro.py .
cp $MAIN/create_root_tree.xml .

n=$1
r=$2

echo "-- Running DDSim..."
ddsim --outputFile /eos/user/e/eneren/sim_lcio_files/pion-shower_$n.slcio --compactFile $lcgeo_DIR/ILD/compact/${SIM_MODEL}/${SIM_MODEL}.xml --steeringFile ddsim_steer_macro.py 

#echo "Converting: LCIO --> root file"
#Marlin create_root_tree.xml --global.LCIOInputFiles=./pion-shower_$n.slcio --MyAIDAProcessor.FileName=pion-shower_$n;

#mkdir -p /eos/user/e/eneren/run_$r 
#mv ./pion-shower_$n.slcio /eos/user/e/eneren/run_$r 
#mv ./pion-shower_$n.root /eos/user/e/eneren/run_$r

