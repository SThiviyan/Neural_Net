//
//  Layer.cpp
//  Neural_Net_Library
//
//  Created by Thiviyan Saravanamuthu on 28.02.21.
//

#include "Layer.hpp"


NN::Layer::Layer(int NeuronNum, int topologyNextElement, Layer* PreviousLayer)
{
    this->NeuronNum = NeuronNum;
    this->topologyNext = topologyNextElement;
    ValMatrix = new Matrix(NeuronNum, 1);
    
    if(PreviousLayer != nullptr)
    {
        this->PreviousLayer = PreviousLayer;
    }
    
    if(topologyNextElement != 0)
    {
        WeightMatrix = new Matrix(topologyNextElement, NeuronNum);
        WeightMatrix->RandomWeightInit();
    }
   
    
}

void NN::Layer::OverrideValMatrix(Matrix *InputValMatrix)
{
 
   for (int row = 0; row < ValMatrix->getRows(); row++) {
            for (int col = 0; col < ValMatrix->getCols(); col++) {
                ValMatrix->operator()(row, col) = InputValMatrix->operator()(row, col);
            }
            
    }
    
    
}

void NN::Layer::OverrideWeightMatrix(Matrix *NewWeights)
{
    //WeightMatrix = nullptr;
    WeightMatrix = new Matrix(topologyNext ,NeuronNum);

    for (int row = 0; row < WeightMatrix->getRows(); row++) {
             for (int col = 0; col < WeightMatrix->getCols(); col++) {
                 WeightMatrix->operator()(row, col) = WeightMatrix->operator()(row, col) + NewWeights->operator()(row, col);
             }
             
     }
}

void NN::Layer::feedforwardValues(ActivationFunctions Ac)
{
    if(PreviousLayer != nullptr)
    {
        Matrix WeightM = PreviousLayer->GetWeightMatrix();
        Matrix ValM = PreviousLayer->GetValMatrix();

        Matrix NewValMatrix = WeightM * ValM;
        NewValMatrix.ActivateNeurons(Ac);
        
        OverrideValMatrix(&NewValMatrix);
        
    }
}



NN::Matrix NN::Layer::GetValMatrix()
{
    return *ValMatrix;
}


NN::Matrix NN::Layer::GetWeightMatrix()
{
    return *WeightMatrix;
}
