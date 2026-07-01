# SLURM-CUDA-JOB
This repository contains scripts for the YouTube video which shows how to use the newly bought  ASUS Ascent GX10 
Armv9-A 128GB 1000GB Blackwell in a SLURM-job. The /home directory is mounted through Ethernet on our local high-performance computing cluster. This causes too long communication times between /home and the machine, which sends the machine in the draining state. In order to avoid such issues the SLURM script was developed where the /temp directory on the node itself is utilized for computations and after when the job is done or failed results are sent back to user's folder from which the script was submitted to SLURM.

