//
//  NN.hpp
//  Neural_Net_Library
//
//  Created by Thiviyan Saravanamuthu on 28.02.21.
//

#ifndef NN_hpp
#define NN_hpp

#include <iostream>
#include <vector>
#include "Layer.hpp"

namespace NN {

    class NN
    {
    public:
        
        //MARK: Constructor/Destructor
        NN(std::vector<int> topology, ActivationFunctions Ac, float LearningRate);
        ~NN();
        
        
        //MARK: Different kinds of excectution for the Network
        
        //Training, Test; Input is all the numbers(separated by layernum)
        void TrainNetwork(std::vector<float> Trainingsset, std::vector<float> Targets);
        void TestNetwork(std::vector<float> Testingset, std::vector<float> Targets);
        
        //Just To run without any backpropagation
        std::vector<float> RunNetwork(std::vector<float> InputSet);
        
        //MARK: The fundamental algorithms to run the network
        void feedforward();
        void backpropagate(std::vector<float> CurrentTargets);
        
        float CalculateCost(std::vector<float> CurrentTargets);
        
        void PrintAll();
        
    private:
        
        std::vector<int> topology;
        int LayerNum;
        Layer* Layers;
        
        float Cost;
        std::vector<std::vector<Matrix>> ErrorGradients;
        int NumBatch;

        float LearningRate;
        ActivationFunctions Ac;
    };


}

#endif /* NN_hpp */
