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


def combine_hdf5(target, output):
    return dsl.ContainerOp(
                    name='Combine hdf5 files',
                    image='ilcsoft/py3lcio:lcio-16',
                    command=[ '/bin/bash', '-c'],
                    arguments=['cd LCIO; source setup.sh; cd .. && \
                                conda init bash; source /root/.bashrc; conda activate root_env && \
                                cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && chmod 600 /tmp/krb5cc_0  && \
                                git clone https://github.com/EnginEren/hgAHCal-ECal.git && cd $PWD/hgAHCal-ECal && \
                                chmod +x hdf5_leftover.sh && ./hdf5_leftover.sh $0 $1', target, output ]
                                            
                
    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount) 


@dsl.pipeline(
    name='ILDEventGen_NestedGAN',
    description='Event Simulation and Reconstruction'
)

def sequential_pipeline():
    """A pipeline with sequential steps."""

    runN ='prod50k'

    
    combine_hdf5('/eos/user/e/eneren/run_' + runN + '/hdf5/',  './mergedData_prod.hdf5')
    
    



   
    
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, __file__ + '.yaml')