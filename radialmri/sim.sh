while read caseidx
do
   echo 'python sim_and_recon.py -c $caseidx -i 0 -d . --spokes 21' #remove dcomp multiplication in simulation, no intensity correction. Thu Jan 28 14:03:21 EST 2021
   python sim_and_recon.py -c $caseidx -i 0 -d . --spokes 21 #remove dcomp multiplication in simulation, no intensity correction. Thu Jan 28 14:03:21 EST 2021
done < sim_BC33.txt
