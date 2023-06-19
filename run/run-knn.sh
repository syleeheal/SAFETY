
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type none --model kNN 

#############################################

python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type feat
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type node 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type unit
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type time 

#############################################

python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type noise --model kNN --noise 0.8
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type noise --model kNN --noise 1.6 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type noise --model kNN --noise 2.4 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type noise --model kNN --noise 3.2 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type noise --model kNN --noise 4.0

#############################################

python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type feat  --mask-prob 0.10 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type feat  --mask-prob 0.20 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type feat  --mask-prob 0.30 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type feat  --mask-prob 0.40 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type feat  --mask-prob 0.50 
python ./SAFETY/main.py --pred joint --time-steps 20 --time-init intermittent --stress-type mask --model kNN --mask-type feat  --mask-prob 0.60


