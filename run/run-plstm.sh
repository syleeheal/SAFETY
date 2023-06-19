python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type none --model P-LSTM --device cuda:0

python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:0 --mask-type feat
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:0 --mask-type node 
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:0 --mask-type unit
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:0 --mask-type time

python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:1 --noise 0.8
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:1 --noise 1.6 
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:1 --noise 2.4 
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:1 --noise 3.2
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:1 --noise 4.0

python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:0 --mask-type feat --mask-prob 0.10 
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:0 --mask-type feat  --mask-prob 0.20
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:1 --mask-type feat  --mask-prob 0.30 
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:1 --mask-type feat  --mask-prob 0.40
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:2 --mask-type feat  --mask-prob 0.50 
python ./SAFETY/main.py --pred task --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:2 --mask-type feat  --mask-prob 0.60

#############################################

python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type none --model P-LSTM --device cuda:2

python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:2 --mask-type feat
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:2 --mask-type node 
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:2 --mask-type unit
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:2 --mask-type time 

python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:3 --noise 0.8
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:3 --noise 1.6
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:3 --noise 2.4
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:3 --noise 3.2
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type noise --model P-LSTM --device cuda:3 --noise 4.0

python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:3 --mask-type feat --mask-prob 0.10 
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:3 --mask-type feat  --mask-prob 0.20 
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:4 --mask-type feat  --mask-prob 0.30 
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:4 --mask-type feat  --mask-prob 0.40 
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:5 --mask-type feat  --mask-prob 0.50 
python ./SAFETY/main.py --pred attack --time-steps 20 --time-init intermittent --stress-type mask --model P-LSTM --device cuda:5 --mask-type feat  --mask-prob 0.60
