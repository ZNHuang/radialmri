while read caseidx
do
   python ../radialmri/sim.py -c $caseidx -d ../sample/kspace --spokes 21 #remove dcomp multiplication in simulation, no intensity correction. Thu Jan 28 14:03:21 EST 2021
done < sim_BC33.txt
