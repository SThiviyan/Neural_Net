//
//  NN.cpp
//  Neural_Net_Library
//
//  Created by Thiviyan Saravanamuthu on 28.02.21.
//

#include "NN.hpp"


NN::NN::NN(std::vector<int> topology, ActivationFunctions Ac, float LearningRate)
{
    this->topology = topology;
    this->LayerNum = topology.size();
    
    this->LearningRate = LearningRate;
    this->Ac = Ac;
    
    this->Layers = (Layer*) malloc(sizeof(Layer) * LayerNum);
    
    for (int n = 0; n < LayerNum; n++) {
        if(n == 0)
        {
            this->Layers[n] = Layer(topology[n], topology[n + 1], nullptr);
            
            Matrix M = Matrix(2, 1);
            M(0, 0) = 0.9f; 
            M(1, 0) = 1.f;
            
            this->Layers[n].OverrideValMatrix(&M);
        }
        else if(n > 0 && n < LayerNum - 1)
        {
            this->Layers[n] = Layer(topology[n], topology[n + 1], &Layers[n - 1]);
        }
        else
        {
            this->Layers[n] = Layer(topology[n], 0, &Layers[n - 1]);
        }
    }
}

NN::NN::~NN()
{
    
}



void NN::NN::TrainNetwork(std::vector<float> Trainingsset, std::vector<float> Targets)
{
    if(Targets.size() % topology[LayerNum - 1] == 0 && Trainingsset.size() % topology[0] == 0)
    {
        
        int NumBatches = Trainingsset.size() / topology[0];
        int BatchSize = topology[0];
        int TargetBatchSize = topology[LayerNum - 1];
        
        for(int n = 1; n <= NumBatches; n++)
        {
            Matrix InputMatrix = Matrix(BatchSize, 1);
            
            std::vector<float> CurrentBatch;
            for (int r = (BatchSize * n) - BatchSize; r < (BatchSize * n); r++) {
                CurrentBatch.push_back(Trainingsset[r]);
            }
            
            for (int z = 0; z < BatchSize; z++) {
                InputMatrix(z, 0) = CurrentBatch[z];
            }
            
            for (int i = 0; i < BatchSize; i++) {
              
            }
            
            
            Layers[0].OverrideValMatrix(&InputMatrix);
            
            feedforward();
            std::cout << "Training Run Nr." << n << ":" << std::endl;
            PrintAll();
            std::cout << std::endl << std::endl << std::endl << std::endl;
            
            std::vector<float> BackpropCurrentTargets;
        
            for (int r = (TargetBatchSize * n) - TargetBatchSize; r < (TargetBatchSize * n); r++) {
                BackpropCurrentTargets.push_back(Targets[r]);
            }
            
               
            backpropagate(BackpropCurrentTargets);
             //PrintAll();
        }
        
    }
    else
    {
        std::cout << "Training Sets and their solutions have to have the same size as topology!";
        return;
    }
}

void NN::NN::TestNetwork(std::vector<float> Testingset, std::vector<float> Targets)
{
    
}

std::vector<float> NN::NN::RunNetwork(std::vector<float> InputSet)
{
    return std::vector<float>();
}



void NN::NN::feedforward()
{
    for(int L = 1; L < LayerNum; L++)
    {
        this->Layers[L].feedforwardValues(this->Ac);
    }
}

void NN::NN::backpropagate(std::vector<float> CurrentTargets)
{
    
}


void NN::NN::PrintAll()
{
    for (int n = 0; n < LayerNum; n++) {
        
        
        if(n == 0)
        {
            Matrix Val = Layers[n].GetValMatrix();

            std::cout << "INPUTLAYER:" << std::endl;
            
            std::cout << std::endl << "ValMatrix:" << std::endl;
            for (int j = 0; j < Val.getRows(); j++) {
                for (int z = 0; z < Val.getCols(); z++) {
                    std::cout << Val(j, z);
                }
                std::cout << std::endl;
            }
            
            Matrix Weight = Layers[n].GetWeightMatrix();
            
            std::cout << std::endl << "WeightMatrix:" << std::endl;
            for (int j = 0; j < Weight.getRows(); j++) {
                for (int z = 0; z < Weight.getCols(); z++) {
                    std::cout << Weight(j, z) << " ";
                }
                std::cout << std::endl;
            }
            
        }
        else if(n > 0 && n < (LayerNum - 1))
        {
            Matrix Val = Layers[n].GetValMatrix();

            
            std::cout << "HIDDENLAYER " << n << ":" << std::endl;
            
            std::cout << std::endl << "ValMatrix:" << std::endl;
            for (int j = 0; j < Val.getRows(); j++) {
                for (int z = 0; z < Val.getCols(); z++) {
                    std::cout << Val(j, z) << " ";
                }
                std::cout << std::endl;
            }
            
            
            Matrix Weight = Layers[n].GetWeightMatrix();
            
            std::cout << std::endl << "WeightMatrix:" << std::endl;
            for (int j = 0; j < Weight.getRows(); j++) {
                for (int z = 0; z < Weight.getCols(); z++) {
                    std::cout << Weight(j, z) << " ";
                }
                std::cout << std::endl;
            }
            
        }
        else
        {
            Matrix Val = Layers[n].GetValMatrix();
            
            std::cout << "OUTPUTLAYER:" << std::endl;
            
            std::cout << std::endl << "ValMatrix:" << std::endl;
            for (int j = 0; j < Val.getRows(); j++) {
                for (int z = 0; z < Val.getCols(); z++) {
                    std::cout << Val(j, z);
                }
                std::cout << std::endl;
            }
        }
        
        
        std::cout << std::endl << std::endl;
        
    }
}
