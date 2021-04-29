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
    this->LayerNum = int(topology.size());
    
    this->LearningRate = LearningRate;
    this->Ac = Ac;
    
    this->Layers = (Layer*) malloc(sizeof(Layer) * LayerNum);
    
    for (int n = 0; n < LayerNum; n++) {
        if(n == 0)
        {
            this->Layers[n] = Layer(topology[n], topology[n + 1], nullptr);
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
        
        int NumBatches = int(Trainingsset.size() / topology[0]);
        this->NumBatch = NumBatches;
        
        int BatchSize = topology[0];
        int TargetBatchSize = topology[LayerNum - 1];
        
        for(int j = 0; j < 10; j++)
        {
            std::cout << "Iteration " << j << std::endl;
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
            
         
            
              Layers[0].OverrideValMatrix(&InputMatrix);
            
              feedforward();
           
                     
              std::vector<float> BackpropCurrentTargets;
          
              for (int r = (TargetBatchSize * n) - TargetBatchSize; r < (TargetBatchSize * n); r++) {
                  BackpropCurrentTargets.push_back(Targets[r]);
                  
              }
                          
              backpropagate(BackpropCurrentTargets);
              
            
        }
            std::cout << std::endl << std::endl << std::endl << std::endl;
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
    /*
    if(Testingset.size() / topology[0]  == Targets.size() / topology[LayerNum - 1])
    {
        for (int n = 0; n < Testingset.size() / topology[0]; n++) {
            
            Matrix InputValMatrix = Matrix(topology[0], 1);
            
            for (int i = 0; i < topology[0]; i++) {
                InputValMatrix(n, 0) = Testingset[n + i];
            }
            
            Layers[0].OverrideValMatrix(&InputValMatrix);
            
            std::cout << "TESTRUN #" << n << ":" << std::endl;
            PrintAll();
            //std::cout << "Error:" << CalculateCost(Targets);
        }
    }
    */
    
    
}

std::vector<float> NN::NN::RunNetwork(std::vector<float> InputSet)
{
    Matrix Inputs = Matrix(InputSet.size(), 1);
    
    for (int n = 0; n < Inputs.getRows(); n++) {
        Inputs(n, 0) = InputSet[n];
    }
    
    Layers[0].OverrideValMatrix(&Inputs);
    
    feedforward();
    
    Matrix Outputs = Layers[LayerNum - 1].GetValMatrix();
    
    std::vector<float> Output_floats;
    for (int n = 0; n < Outputs.getRows(); n++) {
        Output_floats.push_back(Outputs(n, 0));
    }
    
    return Output_floats;
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
    
    
    std::cout << "Cost:" << CalculateCost(CurrentTargets) << std::endl;

    
    //Setting Indeces
    int LayerIndex = LayerNum - 1;

    //Transforming Targets from Vector to Matrix
    Matrix Targets = Matrix(CurrentTargets.size(), 1);
    Targets = CurrentTargets;
    
    
    //Looping backwards through the Network -> for Backprop
    
    Matrix* Prev_Gradient;
    
    for (int n = LayerIndex; n > 0; n--) {
        
        if(n == LayerIndex)
        {
            //MARK: Derivative Cost_Activation
            Matrix Activations = Layers[LayerIndex].GetValMatrix();
            Matrix D_C_A = Targets - Activations;
            
            //MARK: Derivative Activation_Sum
            Matrix D_A_Z = Activations;
            D_A_Z.TakeDerivative(Ac);
       
            //MARK: Gradient calculating via ChainRule
            Matrix Gradient = D_C_A * D_A_Z;
            Gradient.MultiplyByScalar(-1);
            Prev_Gradient = &Gradient;
            //Gradient = Gradient * D_A_Z;
            
            //MARK: Deltaweights -> Calculated with Chain Rule
            // dC/dW = dC/da * da/dz * dz/dw
            //PrevActivations needed because dz/dw = a^l-1
            Matrix PrevActivations = Layers[LayerIndex - 1].GetValMatrix();
            Matrix Deltaweights = (PrevActivations * Gradient).GetTransposedMatrix();
            Deltaweights.MultiplyByScalar(LearningRate);

           
            Layers[n - 1].OverrideWeightMatrix(&Deltaweights);
            
        }
        else
        {
            //MARK: DerivativeCost_Hidden
            //dC/dh = Prev_Gradient * W^T * da/dz        ^T = Transposed
            Matrix Weights = Layers[n].GetWeightMatrix().GetTransposedMatrix();
            Matrix D_A_Z = Layers[n].GetValMatrix();
            D_A_Z.TakeDerivative(Ac);
            Matrix Gradient = D_A_Z * Weights * *Prev_Gradient;
            Prev_Gradient = &Gradient;
            
            //MARK: Deltaweights
            Matrix PrevActivations = Layers[n - 1].GetValMatrix().GetTransposedMatrix();
            Matrix DeltaWeights = Gradient * PrevActivations;
            DeltaWeights.MultiplyByScalar(LearningRate);
            
            Layers[n - 1].OverrideWeightMatrix(&DeltaWeights);
            //int j = 0;
        }
        
        
    }
    
    
}


float NN::NN::CalculateCost(std::vector<float> CurrentTargets)
{
    //feedforward();
    //PrintAll();
    Matrix OutputMatrix = Layers[LayerNum - 1].GetValMatrix();
        
    float Cost = 0;
    
    for (int n = 0; n < OutputMatrix.getRows(); n++) {
        float Temp = CurrentTargets[n] - OutputMatrix(n, 0);
        Temp *= Temp;
        Temp /= 2;
        
        Cost+= Temp;
    }
    
    
    return Cost;
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
