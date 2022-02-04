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


def create_vol():
    return dsl.VolumeOp(
        name="persistent-volume",
        resource_name="my-pvc",
        modes=dsl.VOLUME_MODE_RWO,
        size="15Gi"
    )

def sim(v, pname, rname):
    return dsl.ContainerOp(
                    name='Simulation',
                    image='ilcsoft/ilcsoft-spack:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone https://github.com/EnginEren/hgAHCal-ECal.git  && \
                                cd $PWD/hgAHCal-ECal && chmod +x ./runSimNested.sh && ./runSimNested.sh "$0" "$1" ', pname, rname],
                    pvolumes={"/mnt": v.volume},
                    file_outputs={'lcio_path': '/mnt/lcio_path',
                                   'data': '/mnt/run_'+ rname + '/pion-shower_' + pname + '.slcio',
                    },
    )    

def rec(v, lcio_file, pname, rname):
    return dsl.ContainerOp(
                    name='Reconstruction',
                    image='ilcsoft/ilcsoft-spack:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone https://github.com/EnginEren/hgAHCal-ECal.git && \
                                cd $PWD/hgAHCal-ECal && pwd && \
                                chmod +x ./runRec.sh && ./runRec.sh "$0" "$1" "$2" ', lcio_file, pname, rname ],
                    pvolumes={"/mnt": v.volume},
                    file_outputs={
                            'data': '/mnt/run_'+ rname + '/pion-shower_' + pname + '_REC.slcio'
                    },
    )   


def evaluate(v, lcio_file, inptH5):
    return dsl.ContainerOp(
                    name='Control_Plots',
                    image='ilcsoft/py3lcio:lcio-16',
                    command=[ '/bin/bash', '-c'],
                    arguments=['cd LCIO; source setup.sh; cd .. && \
                                conda init bash; source /root/.bashrc; conda activate root_env && mkdir -p /mnt/plots && \
                                git clone https://github.com/EnginEren/hgAHCal-ECal.git && cd $PWD/hgAHCal-ECal && \
                                python control.py --lcio "$0" --h5file "$1" --nEvents 100 && \
                                cd /mnt/plots/ && touch pion_plots.tar.gz && \
                                tar --exclude=pion_plots.tar.gz -zcvf pion_plots.tar.gz .', lcio_file, inptH5],
                    pvolumes={"/mnt": v.volume},
                    file_outputs = {
                        'data': '/mnt/plots/pion_plots.tar.gz'
                    }
                    
    )   

def convert_hdf5(v, recFile, pname, rname):
    return dsl.ContainerOp(
                    name='hdf5 conversion',
                    image='ilcsoft/py3lcio:lcio-16',
                    command=[ '/bin/bash', '-c'],
                    arguments=['cd LCIO; source setup.sh; cd .. && \
                                conda init bash; source /root/.bashrc; conda activate root_env && \
                                git clone https://github.com/EnginEren/hgAHCal-ECal.git && cd $PWD/hgAHCal-ECal  \
                                && python create_hdf5.py --lcio "$0" --outputR "$1" --outputP "$2" --nEvents 100', recFile, rname, pname],
                    pvolumes={"/mnt": v.volume},
                    file_outputs = {
                        'data': '/mnt/run_'+ rname + '/pion-shower_' + pname + '.hdf5'
                    }
    )   


@dsl.pipeline(
    name='ILDEventGen_NestedGAN',
    description='Event Simulation and Reconstruction'
)

def sequential_pipeline():
    """A pipeline with sequential steps."""
    
    r = create_vol()
    simulation = sim(r, '1', 'testN100')
    #simulation.execution_options.caching_strategy.max_cache_staleness = "P0D"
    inptLCIO = dsl.InputArgumentPath(simulation.outputs['data']) 
    
    reconst = rec(r, inptLCIO, '1', 'testN100')
    inptLCIORec = dsl.InputArgumentPath(reconst.outputs['data'])
    
    hf5 = convert_hdf5(r, inptLCIORec, '1', 'testN100')
    
    inputH5 = dsl.InputArgumentPath(hf5.outputs['data'])
    evaluate(r, inptLCIORec, inputH5)



   
    
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, __file__ + '.yaml')
