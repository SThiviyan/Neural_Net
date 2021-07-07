//
//  DataManager.hpp
//  Neural_Net_Library
//
//  Created by Thiviyan Saravanamuthu on 19.05.21.
//

#ifndef DataManager_hpp
#define DataManager_hpp

#include <iostream>
#include <vector>


    class DataContainer
    {
        
    public:
        
        
        
    private:
        std::vector<float> TestingData;
        std::vector<float> TrainingData;
        
        int instructionSet; //Instructions on how to chop up the data
        
    };



#endif /* DataManager_hpp */
