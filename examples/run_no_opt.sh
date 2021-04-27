#!/bin/bash
counter=1
while [ $counter -le 10 ]
do
   echo "--------------Running test $counter----------------------------"
   ./mnist-cnn -o mnist-cnn.kan -t4 kann-data/mnist-train-?.knd.gz
   ((counter++))
done