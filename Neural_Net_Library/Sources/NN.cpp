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
           
         
              std::cout << "Cost Training Run Nr." << n << ":" ;
            
              std::vector<float> BackpropCurrentTargets;
          
              for (int r = (TargetBatchSize * n) - TargetBatchSize; r < (TargetBatchSize * n); r++) {
                  BackpropCurrentTargets.push_back(Targets[r]);
               }
            
              std::cout << CalculateCost(BackpropCurrentTargets) << std::endl;

              //PrintAll();
              
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
    //std::cout << CalculateCost(CurrentTargets) << std::endl;
    
    std::vector<Matrix> CurrentErrorGradients;
    
    for (int n = LayerNum - 1; n > 0; n--) {
       
        if(n == LayerNum - 1)
        {
            
            //MARK: Calculating Gradient of last Layer
            
            Matrix Targets = Matrix(Layers[n].GetValMatrix().getRows(), 1);
            Matrix Activations = Layers[n].GetValMatrix();
            
            for (int n = 0; n < Targets.getRows(); n++) {
                Targets(n, 0) = CurrentTargets[n];
            }
        
            Matrix DerivativeCostActivation = Activations - Targets;
    
            Matrix PreWeight = Layers[n - 1].GetWeightMatrix();
            Matrix PreActivations = Layers[n - 1].GetValMatrix();
            Matrix DerivativeActivationSum = PreWeight * PreActivations;
                
            switch (this->Ac) {
                case SIGMOID:
                    DerivativeActivationSum.ActivateNeurons(D_SIGMOID);
                    break;
                case RELU:
                    DerivativeActivationSum.ActivateNeurons(D_RELU);
                    break;
                default:
                    break;
            }
            
            Matrix Gradient = DerivativeCostActivation * DerivativeActivationSum;
            
            CurrentErrorGradients.push_back(Gradient);
            
            //this->ErrorGradients[(LayerNum - 1) - n].push_back(Gradient);
            
        }
        
        else
        {
            //MARK: Calculating Gradient in inner Layers
            
            Matrix Pre_Gradient = CurrentErrorGradients[(LayerNum-1) - n - 1];
            
            Matrix CorrespondingWeight = Layers[n].GetWeightMatrix().GetTransposedMatrix();
            
            Matrix DerivativeCostHidden = CorrespondingWeight * Pre_Gradient;
            
       
            
            
            
            Matrix PreWeight = Layers[n - 1].GetWeightMatrix();
            Matrix PreActivations = Layers[n - 1].GetValMatrix();
            Matrix DerivativeActivationSum = PreWeight * PreActivations;
           
            switch (this->Ac) {
                case SIGMOID:
                    DerivativeActivationSum.ActivateNeurons(D_SIGMOID);
                    break;
                case RELU:
                    DerivativeActivationSum.ActivateNeurons(D_RELU);
                    break;
                default:
                    break;
            }
            
            Matrix Gradient = DerivativeCostHidden * DerivativeActivationSum;

            CurrentErrorGradients.push_back(Gradient);
            //this->ErrorGradients[(LayerNum - 1) - n].push_back(Gradient);

        }
        
    }
    
    this->ErrorGradients.push_back(CurrentErrorGradients);
    
    int AverageCost = 0;
    AverageCost += CalculateCost(CurrentTargets);
    
    if(ErrorGradients.size() == NumBatch)
    {
        AverageCost = AverageCost / NumBatch;
        
        
        std::vector<Matrix> AverageGradients;
        for (int n = 0; n < ErrorGradients.size(); n++) {
            for (int j = 0; j < ErrorGradients[n].size(); j++) {
                
                Matrix Gradient = ErrorGradients[n][j];

                if(n == 0)
                {
                    AverageGradients.push_back(Gradient);
                }
                else if(n > 0 && n < (NumBatch - 1))
                {
                    AverageGradients[j] = AverageGradients[j] + Gradient;
                    
                    
                }
                else
                {
                    AverageGradients[j] = AverageGradients[j] + Gradient;
                    AverageGradients[j].DivideByScalar(NumBatch);
                }
                
            
        }
        
        
        ErrorGradients.clear();
        AverageCost = 0;
       }
    
       
       for (int L = LayerNum - 1; L > 0; L--)
       {
           Matrix CurrentGradient = AverageGradients[(LayerNum - 1) - L];
           Matrix CorrespondingActivations = Layers[L - 1].GetValMatrix().GetTransposedMatrix();
           
           Matrix DeltaWeights = CurrentGradient * CorrespondingActivations;
           //DeltaWeights.MultiplyByScalar(LearningRate);
       
           Layers[L - 1].OverrideWeightMatrix(&DeltaWeights);
           
           
       }
        
        
        
    }
}


float NN::NN::CalculateCost(std::vector<float> CurrentTargets)
{
    
    Matrix OutputMatrix = Layers[LayerNum - 1].GetValMatrix();
        
    Cost = 0;
    
    std::vector<float> CostArray;
    for (int n = 0; n < OutputMatrix.getRows(); n++) {
        float Temp = OutputMatrix(n, 0) - CurrentTargets[n];
        Temp *= Temp;
        Temp = 0.5 * Temp;
        CostArray.push_back(Temp);
    }
    
    for (int n = 0; n < OutputMatrix.getRows(); n++) {
        Cost += CostArray[n];
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
