#!/usr/bin/env python3
import uproot
import datetime
import logging
import hist

from coffea import util, processor
from coffea.nanoevents import NanoAODSchema

from vll.processor import VLLProcessor
from vll.utils.crossSections import lumis, crossSections, nevents

import time
import sys
import os

import argparse

from samples.fileLists.MC_2018 import filelist2018 as filelistMC
from samples.fileLists.Data_2018 import dataSingleMu2018 as filelistMu
from samples.fileLists.Data_2018 import dataSingleEle2018 as filelistEle

# To fix - add all in same file
filelist = filelistEle

# Define mapping for running on condor
mc_group_mapping = {
    "Signal": [key for key in filelist if "VLL" in key],
    "MCTTbar1l": ["TTbarPowheg_Semilept", "TTbarPowheg_Hadronic"],
    "MCTTbar2l": ["TTbarPowheg_Dilepton"],
    "TTV": ["TTW","TTZ", "TTHToNonBB","TTHToBB"],
    "TTVV": ["TTHH","TTWW","TTZZ","TTZH","TTWZ","TTWH","TTTT"],
    "MCZJets": [key for key in filelist if "DY" in key],
    "DiBoson": ["WW","ZZ","WZ"],
    "TriBoson": ["WWW","ZZZ","WWZ","WZZ"],
    "MCSingleTop": [key for key in filelist if "ST" in key],
}

mc_nonother = {key for group in mc_group_mapping.values() for key in group}

mc_group_mapping["MCOther"] = [
    key for key in filelist if (not key in mc_nonother) and (not "Data" in key)
]
mc_group_mapping["MCAll"] = [
    key for group in mc_group_mapping.values() for key in group
]

def move_X509():
    try:
        _x509_localpath = (
            [
                line
                for line in os.popen("voms-proxy-info").read().split("\n")
                if line.startswith("path")
            ][0]
            .split(":")[-1]
            .strip()
        )
    except Exception as err:
        raise RuntimeError(
            "x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
        ) from err
    _x509_path = f'/scratch/{os.environ["USER"]}/{_x509_localpath.split("/")[-1]}'
    os.system(f"cp {_x509_localpath} {_x509_path}")
    return os.path.basename(_x509_localpath)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
        level=logging.WARNING,
    )

    parser = argparse.ArgumentParser(
        description="Batch processing script for vll analysis"
    )
    parser.add_argument(
        "mcGroup",
        choices=list(mc_group_mapping) + ["Data"],
        help="Name of process to run",
    )
    parser.add_argument(
        "--era",
        choices=["2018","2017","2016pre","2016post"],
        default="2018",
        help="Era to run over",
    )
    parser.add_argument("--chunksize", type=int, default=10000, help="Chunk size")
    parser.add_argument("--maxchunks", type=int, default=None, help="Max chunks")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--outdir", type=str, default="Outputs", help="Where to put the output files")
    parser.add_argument(
        "--batch", action="store_true", help="Batch mode (no progress bar)"
    )
    parser.add_argument(
        "-e",
        "--executor",
        choices=["local", "wiscjq", "debug"],
        default="local",
        help="How to run the processing",
    )
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    tstart = time.time()

    print("Running mcGroup {}".format(args.mcGroup))

    if args.executor == "local":
        if args.workers > 4:
            raise RuntimeError("You probably shouldn't run more than 4 cores locally")
        executor = processor.FuturesExecutor(
            workers=args.workers, status=not args.batch
        )
    elif args.executor == "debug":
        executor = processor.IterativeExecutor(status=not args.batch)
    elif args.executor == "wiscjq":
        from distributed import Client
        from dask_jobqueue import HTCondorCluster

        if args.workers == 1:
            print("Are you sure you want to use only one worker?")

        os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
        _x509_path = move_X509()

        cluster = HTCondorCluster(
            cores=1,
            memory="2 GB",
            disk="1 GB",
            death_timeout = '60',
            job_extra_directives={
                "+JobFlavour": '"tomorrow"',
                "log": "dask_job_output.$(PROCESS).$(CLUSTER).log",
                "output": "dask_job_output.$(PROCESS).$(CLUSTER).out",
                "error": "dask_job_output.$(PROCESS).$(CLUSTER).err",
                "should_transfer_files": "yes",
                "when_to_transfer_output": "ON_EXIT_OR_EVICT",
                "+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest"',
                "Requirements": "HasSingularityJobStart",
                "InitialDir": f'/scratch/{os.environ["USER"]}',
                "transfer_input_files": f'{os.environ["EXTERNAL_BIND"]}/.env,{_x509_path},{os.environ["EXTERNAL_BIND"]}/vll'
            },
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",
                f"export X509_USER_PROXY={_x509_path}",
            ]
        )
        cluster.adapt(minimum=1, maximum=args.workers)
        executor = processor.DaskExecutor(client=Client(cluster), status=not args.batch)

    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=args.maxchunks,
        skipbadfiles=True,
        xrootdtimeout=300,
    )

    if args.mcGroup == "Data":
        print('Data')
        job_fileset = {key: filelist[key] for key in filelist if "Data" in key}
        output = runner(
            job_fileset,
            treename="Events",
            processor_instance=VLLProcessor(isMC=False,era=args.era),
        )
    else:
        print('MC')
        job_fileset = {key: filelist[key] for key in mc_group_mapping[args.mcGroup]}
        output = runner(
            job_fileset,
            treename="Events",
            processor_instance=VLLProcessor(isMC=True,era=args.era),
        )

        # Compute original number of events for normalization
        #for dataset_name, dataset_files in job_fileset.items():
        #    # Calculate luminosity scale factor
        #    lumi_sf = (
        #        crossSections[dataset_name]
        #        * lumis[args.era]
        #        / nevents[args.era][dataset_name]
        #    )

        #    print(dataset_name,":",nevents[args.era][dataset_name])

        #   for key, obj in output[dataset_name].items():
        #        if isinstance(obj, hist.Hist):
        #            obj *= lumi_sf

    elapsed = time.time() - tstart
    print(f"Total time: {elapsed:.1f} seconds")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(args.outdir, f"output_{args.mcGroup}_run{timestamp}.coffea")
    util.save(output, outfile)
    print(f"Saved output to {outfile}")
