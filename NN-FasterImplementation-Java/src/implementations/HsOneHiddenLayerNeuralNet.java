package implementations;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Scanner;

import interfaces.NeuralNetInterface;

public class HsOneHiddenLayerNeuralNet implements NeuralNetInterface {
	
	int bias = 1;
	
	int argNumInputs;
	int argNumHidden;
	double argLearningRate;
	double argMomentumTerm;
	boolean binary;

    HiddenNeuran hiddenLayer[];

    
    
	public HsOneHiddenLayerNeuralNet ( 
			int numOfInputs,
            int numOfHiddens,
            double learningRate,
            double momentumTerm,
            boolean binary) {

		this.argNumInputs = numOfInputs;
		this.argNumHidden = numOfHiddens;
		this.argLearningRate = learningRate;
		this.argMomentumTerm = momentumTerm;
		this.binary = binary;
		
		hiddenLayer = new HiddenNeuran[argNumHidden + 1];
	}

	@Override
	public double outputFor(double[] X) {

		if(binary){
			double output = customSigmoid(hiddenLayer[0].outputWeight * bias);
			
			for( int i = 1; i < hiddenLayer.length; i++) {
				double activation = customSigmoid(hiddenLayer[i].calculateOutput(X, bias) );
				hiddenLayer[i].currentOutput = activation;
				output += activation;
			}

			return customSigmoid(output);
		}
		
		else {
			double output = sigmoid(hiddenLayer[0].outputWeight * bias);

			for( int i = 1; i < hiddenLayer.length; i++) {
				double activation = sigmoid(hiddenLayer[i].calculateOutput(X, bias) );
				hiddenLayer[i].currentOutput = activation;
				output += activation;
			}

			return sigmoid(output);
		}
	}

	@Override
	public double train(double[] X, double argValue) {
		
		double output = outputFor(X);
		
		
		//calculate weight correction term for hidden to output edges (outputDelta) 
		double errorInfoTerm_output;
		if(binary) {
			errorInfoTerm_output = ( argValue - output ) * ( output * ( 1 - output ) );  
		}
		else {
			errorInfoTerm_output = ( argValue - output) * 0.5 * ( 1 + output ) * ( 1 - output );
		}
		
		//for bias node in hidden
		hiddenLayer[0].outputDelta = argLearningRate * errorInfoTerm_output * bias;
		
		for (int i = 1; i < hiddenLayer.length; i++) {
			hiddenLayer[i].outputDelta = argLearningRate * errorInfoTerm_output * hiddenLayer[i].currentOutput;
		}
		
		
		//calculate the error (infoTerm) for each node in hidden layer
		for (HiddenNeuran n : hiddenLayer) {
			
			n.deltaInputSum = errorInfoTerm_output * n.outputWeight;
			
			if(binary) {
				n.infoTerm = n.deltaInputSum * n.currentOutput * ( 1 - n.currentOutput ) ;
			}
			else {
				n.infoTerm = n.deltaInputSum * 0.5 * ( 1 + n.currentOutput ) * ( 1 - n.currentOutput );
			}
		}

		
		//Compute weight correction term for input to hidden edges (inputDeltas)
		for (HiddenNeuran n : hiddenLayer) {
			//for bias
			n.inputDeltas[0] = argLearningRate * n.infoTerm * bias;
			
			for(int j = 1; j < X.length+1; j++) {
				n.inputDeltas[j] = argLearningRate * n.infoTerm * X[j-1];
			}
		}
		
		
		
		/* update the weights between the hidden layer and the output */
		for(HiddenNeuran n : hiddenLayer) {
			n.nextOutputWeight = n.outputWeight + argMomentumTerm * n.changeInOutputWeight + n.outputDelta;
			
			n.changeInOutputWeight = n.outputWeight - n.prevOutputWeight;
			
			n.prevOutputWeight = n.outputWeight;
			n.outputWeight = n.nextOutputWeight;
		}
		
		/* Update input to hidden weights */      
		for(int i = 1; i < hiddenLayer.length; i++) {
			HiddenNeuran n = hiddenLayer[i];
			
			for(int j = 0; j < X.length+1; j++) {
				n.nextInputWeights[j] = n.inputWeights[j] + argMomentumTerm * n.changeInInputWeights[j] + n.inputDeltas[j];
				
				n.changeInInputWeights[j] = n.inputWeights[j] - n.prevInputWeights[j];
				
				n.prevInputWeights[j] = n.inputWeights[j];
				n.inputWeights[j] = n.nextInputWeights[j];
			}
		}
		

		return (argValue - output);
	}

	@Override
	public void save(File argFile) {
		try {
        	FileOutputStream is = new FileOutputStream(argFile);
        	OutputStreamWriter osw = new OutputStreamWriter(is);    
        	
        	Writer w = new BufferedWriter(osw);

      		for (HiddenNeuran n : hiddenLayer) {
      			w.write( Double.toString(n.outputWeight)+ " ");	
      			for(int i =0; i < n.inputWeights.length; i++)
      				w.write( Double.toString(n.inputWeights[i])+ " ");	
    		}

        	w.close();		
		} catch (IOException e) {
            System.err.println("Problem writing to the file statsTest.txt");
        }
	}

	@Override
	public void load(String argFileName) throws IOException {
		
		Charset ENCODING = StandardCharsets.UTF_8;
		Path fFilePath = Paths.get(argFileName);
		
		for(HiddenNeuran n : hiddenLayer){
			
			ArrayList<Double> weights = new ArrayList<>();

			try{
				Scanner scanner =  new Scanner(fFilePath,ENCODING.name());
				scanner.useDelimiter(" ");

				while ( scanner.hasNext() ) {
					weights.add(scanner.nextDouble());
				}
			} catch(IOException e) {System.out.println("could not open file to load");}

			n.setWeights(weights);
		}
	}

	@Override
	public double sigmoid(double x) {
		return (  2 / ( 1 + Math.exp( -x ) ) ) -1;
	}

	@Override
	public double customSigmoid(double x) {
		return ( 1 / (1 + Math.exp(-x) ) );
	}

	@Override
	public void initializeWeights() {

        for(int i=0; i< argNumHidden + 1; i++){
            hiddenLayer[i] = new HiddenNeuran(2, (Math.random() * 1) - 0.5);
        }
	}

	@Override
	public void zeroWeights() {
		
        for(int i=0; i< argNumHidden + 1; i++){
            hiddenLayer[i] = new HiddenNeuran(2, 0);
        }
	}

}
