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
    topology.push_back(3);
    topology.push_back(1);
    
    NN::NN NeuralNet = NN::NN(topology,
                              NN::SIGMOID,
                              0.1);
    
    
    std::vector<float> Trainingset = {1, 1, 0, 0, 1, 0, 0, 1};
    std::vector<float> Targets = {1, 0, 0, 0};
    
    NeuralNet.TrainNetwork(Trainingset, Targets);
    
    /*
    std::vector<float> Testingset = {0,0};
    
    std::vector<float> Output = NeuralNet.RunNetwork(Testingset);
        
    NeuralNet.PrintAll();
 
    Testingset.clear();
    Testingset.push_back(1);
    Testingset.push_back(1);
    
    Output = NeuralNet.RunNetwork(Testingset);
    
    NeuralNet.PrintAll();

    
    Testingset.clear();
    Testingset.push_back(1);
    Testingset.push_back(0);
     
    Output = NeuralNet.RunNetwork(Testingset);
    
    NeuralNet.PrintAll();


     
    Testingset.clear();
    Testingset.push_back(0);
    Testingset.push_back(1);
    
    Output = NeuralNet.RunNetwork(Testingset);
    
    NeuralNet.PrintAll();

    */
    
    NN::Matrix Test = NN::Matrix(1, 1);
    Test(0, 0) = 0.75136507f;
    Test.ActivateNeurons(NN::D_SIGMOID);
    
    std::cout << Test(0, 0);
    
    return 0;
}
