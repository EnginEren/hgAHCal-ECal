#!/usr/bin/env python3
# Copyright 2019 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tarfile import RECORDSIZE
import kfp
from kfp import dsl
from kfp.components import InputPath, InputTextFile, InputBinaryFile, OutputPath, OutputTextFile, OutputBinaryFile
from kubernetes import client as k8s_client

eos_host_path = k8s_client.V1HostPathVolumeSource(path='/var/eos')
eos_volume = k8s_client.V1Volume(name='eos', host_path=eos_host_path)
eos_volume_mount = k8s_client.V1VolumeMount(name=eos_volume.name, mount_path='/eos')

krb_secret = k8s_client.V1SecretVolumeSource(secret_name='krb-secret')
krb_secret_volume = k8s_client.V1Volume(name='krb-secret-vol', secret=krb_secret)
krb_secret_volume_mount = k8s_client.V1VolumeMount(name=krb_secret_volume.name, mount_path='/secret/krb-secret-vol')


def sim(pname, rname):
    return dsl.ContainerOp(
                    name='Simulation',
                    image='ilcsoft/ilcsoft-spack:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone https://github.com/EnginEren/hgAHCal-ECal.git && whoami && \
                                cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && \
                                chmod 600 /tmp/krb5cc_0 &&  \
                                cd $PWD/hgAHCal-ECal && chmod +x ./runSimNestedEOS.sh && ./runSimNestedEOS.sh "$0" "$1" ', pname, rname],
                    file_outputs={
                        'metadata': '/mnt/lcio_path'
                    }

    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)    

def rec(lcio_file, pname, rname):
    return dsl.ContainerOp(
                    name='Reconstruction',
                    image='ilcsoft/ilcsoft-spack:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone https://github.com/EnginEren/hgAHCal-ECal.git && \
                                cd $PWD/hgAHCal-ECal && \
                                cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && \
                                chmod 600 /tmp/krb5cc_0 &&  \
                                chmod +x ./runRec.sh && ./runRec.sh "$0" "$1" "$2" ', lcio_file, pname, rname ],
                    file_outputs={
                        'metadata': '/mnt/lcio_rec_path'
                    }    
    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)   


def evaluate(lcio_file, inptH5):
    return dsl.ContainerOp(
                    name='Control_Plots',
                    image='ilcsoft/py3lcio:lcio-16',
                    command=[ '/bin/bash', '-c'],
                    arguments=['cd LCIO; source setup.sh; cd .. && \
                                conda init bash; source /root/.bashrc; conda activate root_env && mkdir -p /mnt/plots && \
                                git clone https://github.com/EnginEren/hgAHCal-ECal.git && cd $PWD/hgAHCal-ECal && \
                                cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && chmod 600 /tmp/krb5cc_0 && \
                                python controlEOS.py --lcio "$0" --h5file "$1" --nEvents 1000 && \
                                cd /mnt/plots/ && touch pion_plots.tar.gz && \
                                tar --exclude=pion_plots.tar.gz -zcvf pion_plots.tar.gz .', lcio_file, inptH5],
                    file_outputs = {
                        'data': '/mnt/plots/pion_plots.tar.gz'
                    }
                    
    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)   

def convert_hdf5(recFile, pname, rname):
    return dsl.ContainerOp(
                    name='hdf5 conversion',
                    image='ilcsoft/py3lcio:lcio-16',
                    command=[ '/bin/bash', '-c'],
                    arguments=['cd LCIO; source setup.sh; cd .. && \
                                conda init bash; source /root/.bashrc; conda activate root_env && \
                                git clone https://github.com/EnginEren/hgAHCal-ECal.git && cd $PWD/hgAHCal-ECal && \
                                cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && chmod 600 /tmp/krb5cc_0 \
                                && python create_hdf5EOS.py --lcio "$0" --outputR "$1" --outputP "$2" --nEvents 1000', recFile, rname, pname],
                    file_outputs={
                        'metadata': '/mnt/hdf5_path'
                    }                
                
    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)   


def combine_hdf5(flist, output):
    return dsl.ContainerOp(
                    name='Combine hdf5 files',
                    image='ilcsoft/py3lcio:lcio-16',
                    command=[ '/bin/bash', '-c'],
                    arguments=['cd LCIO; source setup.sh; cd .. && \
                                conda init bash; source /root/.bashrc; conda activate root_env && \
                                git clone https://github.com/EnginEren/hgAHCal-ECal.git && cd $PWD/hgAHCal-ECal && \
                                cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && chmod 600 /tmp/krb5cc_0 \
                                && python combineEOS.py --input $0 $1 $2 --output $3', flist[0], flist[1], flist[2] , output]            
                
    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)  



@dsl.pipeline(
    name='ILDEventGen_NestedGAN',
    description='Event Simulation and Reconstruction'
)

def sequential_pipeline():
    """A pipeline with sequential steps."""
    

    runN = 'prod10k_20GeV'
    simBase = sim(0, runN)
    inptLCIOb = dsl.InputArgumentPath(simBase.outputs['metadata']) 
    
    reconstBase = rec(inptLCIOb, 0, runN)
    inptLCIORecb = dsl.InputArgumentPath(reconstBase.outputs['metadata'])
    
    hf5b = convert_hdf5(inptLCIORecb, 0, runN)
    inptH5b = dsl.InputArgumentPath(hf5b.outputs['metadata'])
    evaluate(inptLCIORecb, inptH5b)
    
    ## submit many jobs without control plots
    
    h5outs = []
    for i in range(1,10):
        runN = 'prod10k_20GeV'
        simulation = sim(str(i), runN)
        #simulation.execution_options.caching_strategy.max_cache_staleness = "P0D"
        inptLCIO = dsl.InputArgumentPath(simulation.outputs['metadata']) 
        
        reconst = rec(inptLCIO, str(i), runN)
        #reconst.execution_options.caching_strategy.max_cache_staleness = "P0D"
        inptLCIORec = dsl.InputArgumentPath(reconst.outputs['metadata'])
        
        hf5 = convert_hdf5(inptLCIORec, str(i), runN)
        a = dsl.InputArgumentPath(hf5.outputs['metadata'])
        h5outs.append(a)

    
    combine_hdf5(h5outs, '/eos/user/e/eneren/run_'+runN + '/mergedData_prod10k.hdf5')
    #print (h5outs)
    



   
    
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, __file__ + '.yaml')
