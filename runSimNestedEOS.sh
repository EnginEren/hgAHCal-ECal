#!/bin/bash

source /home/init_ilcsoft.sh
source /spack/share/spack/setup-env.sh

export SIM_MODEL=ILD_l5_v02
export MAIN=$(echo $PWD)

git clone --branch v02-02 https://github.com/iLCSoft/ILDConfig.git
cd ILDConfig/StandardConfig/production


cp $MAIN/pionGun.mac . 
cp $MAIN/ddsim_steer_macro.py .


n=$1
r=$2

export EOS_home=/eos/user/e/eneren
mkdir -p /eos/user/e/eneren/run_$r

echo "-- Running DDSim..."
ddsim --outputFile $EOS_home/run_$r/pion-shower_$n.slcio --compactFile $lcgeo_DIR/ILD/compact/${SIM_MODEL}/${SIM_MODEL}.xml --steeringFile ddsim_steer_macro.py 


