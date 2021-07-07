//
//  main.cpp
//  Neural_Net_Library
//
//  Created by Thiviyan Saravanamuthu on 28.02.21.
//

#include <iostream>
#include "NN.hpp"

int main()
{
    
    std::vector<int> topology;
    topology.push_back(2);
    topology.push_back(5);
    topology.push_back(1);
    
    NN::NN NeuralNet = NN::NN(topology,
                              NN::SIGMOID,
                              0.5);
    
    
    std::vector<float> Trainingset = {1, 1, 0, 0, 1, 0, 0, 1};
    std::vector<float> Targets = {0, 0, 1, 1};
    
    NeuralNet.TrainNetwork(Trainingset, Targets);

    
    
    return 0;
}
