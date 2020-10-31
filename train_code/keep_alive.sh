#!/bin/bash

for i in `seq 0 72`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  google-chrome https://colab.research.google.com/drive/1TVqVZU0Ty9f_Yb9JpW95NUhHPsUad3qH#scrollTo=ZAWe7NFEMmaB
  sleep 600
done
