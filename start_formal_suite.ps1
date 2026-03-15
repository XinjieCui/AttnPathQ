$ErrorActionPreference = "Stop"

wsl -d Ubuntu-22.04 -- bash -lc "bash /home/cxj/experiments/vit_quant_5ideas/launch_formal_suite.sh > /home/cxj/experiments/vit_quant_5ideas/results/logs/suite_master.out 2>&1"
