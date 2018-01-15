package implementations;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


public class main {

	public static void main(String[] args) throws IOException{
		
		int trial = 0;
		boolean binary = false;
		double momentumTerm = 0.9;
		double learningRate = 0.2;
		
		
		double binary_input[][] = { {0,0}, {0,1}, {1,0}, {1,1} };
		double bipolar_input[][] = { {-1,-1}, {-1,1}, {1,-1}, {1,1} };

		double expectedBinaryOutput[] = { 0, 1, 1, 0 };     
		double expectedBipolarOutput[] = { -0.9, 0.9, 0.9, -0.9 }; 
		
		HsOneHiddenLayerNeuralNet nn = new HsOneHiddenLayerNeuralNet(2, 4, learningRate, momentumTerm, binary);
	
		nn.initializeWeights();
		
		FileWriter fileWriter = new FileWriter("hs" + new Integer(trial).toString() +".csv", true);
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
        
        double totalError = 1000;
        int epoch = 1;
        
        while(totalError > 0.05){
        	
        	double epochsEr = 0;
        	double input[][];
        	double expectedOutput[];
        	
        	if(binary){
        		input = binary_input;
        		expectedOutput = expectedBinaryOutput;
        	}
        	else{
        		input = bipolar_input;
        		expectedOutput = expectedBipolarOutput;
        	}
        	
        	for(int i = 0; i < 4; i++){
    			double output = nn.outputFor(input[i]);
    			epochsEr += Math.pow(nn.train(input[i], expectedOutput[i]), 2);
    		}
    		
    		totalError += epochsEr / 2;
    		
			if( epoch++ % 10 == 0 )
			{
				System.out.println( epoch + "," + totalError ); 
				bufferedWriter.write( epoch + "," + totalError );
				bufferedWriter.newLine();
			}  
        }
           
	}

}
