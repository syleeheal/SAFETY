python ./train/main.py --pred y_int --stress-type none --model SAFETY

python ./train/main.py --pred y_int --stress-type noise --model SAFETY --noise 0.8
python ./train/main.py --pred y_int --stress-type noise --model SAFETY --noise 1.6 
python ./train/main.py --pred y_int --stress-type noise --model SAFETY --noise 2.4 
python ./train/main.py --pred y_int --stress-type noise --model SAFETY --noise 3.2 
python ./train/main.py --pred y_int --stress-type noise --model SAFETY --noise 4.0

python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.10 
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.20
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.30 
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.40
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.50 
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.60

python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.50
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type node --mask-prob 0.50
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type unit --mask-prob 0.50
python ./train/main.py --pred y_int --stress-type mask --model SAFETY --mask-type time --mask-prob 0.50

#############################################

python ./train/main.py --pred y_atk --stress-type none --model SAFETY

python ./train/main.py --pred y_atk --stress-type noise --model SAFETY --noise 0.8
python ./train/main.py --pred y_atk --stress-type noise --model SAFETY --noise 1.6 
python ./train/main.py --pred y_atk --stress-type noise --model SAFETY --noise 2.4 
python ./train/main.py --pred y_atk --stress-type noise --model SAFETY --noise 3.2 
python ./train/main.py --pred y_atk --stress-type noise --model SAFETY --noise 4.0

python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.10 
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.20
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.30 
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.40
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.50 
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.60

python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type feat --mask-prob 0.50
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type node --mask-prob 0.50
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type unit --mask-prob 0.50
python ./train/main.py --pred y_atk --stress-type mask --model SAFETY --mask-type time --mask-prob 0.50


