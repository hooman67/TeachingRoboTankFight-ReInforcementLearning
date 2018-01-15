package implementations;

import java.util.ArrayList;

public class HiddenNeuran {

	int numInputs;
	double currentInputSum;
	double currentOutput;
	
	
	double inputWeights[];
	double nextInputWeights[];
	double prevInputWeights[];
	double changeInInputWeights[];
	
	
	double outputWeight;
	double nextOutputWeight;
	double prevOutputWeight;
	double changeInOutputWeight;
	
	
	double inputDeltas[];
	double outputDelta;

    double deltaInputSum;
    double infoTerm;
    
    
    public HiddenNeuran(int argNumInputs, double initialWeightValue) {
    	
    	this.numInputs = argNumInputs + 1;
    	
    	
    	outputWeight = initialWeightValue;
        nextOutputWeight = prevOutputWeight = changeInOutputWeight = 0;
        
        
        inputWeights = nextInputWeights = prevInputWeights = changeInInputWeights = inputDeltas =
        		new double[numInputs];
    	
    	for(int i =0; i < numInputs; i++) {
    		inputWeights[i] = initialWeightValue;
    		nextInputWeights[i] = prevInputWeights[i] = changeInInputWeights[i] = inputDeltas[i] = 0;
    	}
    	
    	
    	currentInputSum = currentOutput = deltaInputSum = infoTerm = 0;
    }
    
    double calculateOutput( double[ ] X, double bias ) {
 //   	if(X.length != inputWeights.length -1 ) {
   // 		System.out.println("getOutputForNueran: Wrong input length!");
    //		return -1000;
    	//}
    	
    	double output = 0;

    	output = bias * inputWeights[0];

    	for(int i = 0; i < X.length; i++) {
    		output += X[i] * inputWeights[i+1];
    	}

    	output = currentInputSum*outputWeight;


    	return output;
    }
    
    public void setWeights(ArrayList<Double> weights) {
    	outputWeight = weights.get(0);
    	
    	for(int i = 0; i < inputWeights.length; i++){
    		inputWeights[i] = weights.get(i+1);
    	}
    	
    }
    

}
