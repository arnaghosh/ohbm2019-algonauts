# OHBM Hackathon 2019 - Algonauts Challenge
OHBM Hackathon 2019 project repository for the Algonauts challenge

To do list :-
+ [X] Write functions for loading data and labels
+ [X] Write model class
+ [X] Visualize the variance in RDMs across subjects - will help take decisions for subject level model
+ [X] Train a model's final feature layer to explain late visual cortex - fMRI --> running
+ [ ] Train a model's initial feature layers to explain early visual cortex - fMRI
+ [ ] Train a model's final feature layer to explain late visual cortex activity - MEG
+ [ ] Train a model's initial feature layers to explain early visual cortex activity - MEG
+ [ ] Train a model jointly to optimize for early and late activity

Major issues :-
+ [ ] How to tackle subject variance? Do we treat them as noisy labels? Do we train separate models and then average predictions?

Some ideas :-
* Train each model for eash subject, generate predictions and average
* Average train dataset for all subejets and train single model
* Train Siamese networks with pretrained CNNs to optimize IT/EVC RDMs (rep. dissimilarity matrix)
* Could also try to jointly optimize both IT and EVC --> train on Pearson similarity for EVC on feats at a lower Layer and IT for higher layer
