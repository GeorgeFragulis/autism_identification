format compact;clear;close all;clc
%%  GFF  Run the script for Web Browse dataset

warning("off")
disp("Running Model Please wait :)!")

trainingData = readtable('Web_browse_all.csv');
inputTable = trainingData;
trainClassifier(trainingData)

