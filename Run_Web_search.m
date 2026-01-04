format compact;clear;close all;clc
%%  GFF  Run the script for Web search dataset

warning("off")
disp("Running Model Please wait :)!")

trainingData = readtable('Web_search_all.csv');
inputTable = trainingData;
trainClassifierSearch(trainingData)