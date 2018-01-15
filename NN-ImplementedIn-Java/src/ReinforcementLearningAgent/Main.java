package nnClassAssignment;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;

import activations.BipolarSigmoidActivation;
import activations.SigmoidActivation;

public class Main {
	public static void main(String[] args) throws IOException {
		
		NuralNetwork nn = new NuralNetwork(2, 1, 4, new BipolarSigmoidActivation(), 0.05, 0); 
		//bipolar moment >= 0.134 always converges. As momentum goes lower the probability of convergence goes down too
		//Try 0.01; it sometimes converges.  Alpha 0.05 works with momentum of 0 too.
		
	    double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	    double[][] outputs = {{-0.8}, {0.8}, {0.8}, {-0.8}};
	  //  double[][] outputs = {{0}, {1}, {1}, {0}};
	    
	    saveTrainingErrors(nn.train(inputs, outputs, 0.05));//0.0001

		nn.hsSave();
		
//		nn.load("HsTrainedXORNetwork-1476827410960.HS");
		
		
        System.out.println("Testing hsTrained XOR neural network");

        nn.setInputs(new double[]{0, 0});
        System.out.println("0 XOR 0: " + (nn.getOutput()[0]));

        nn.setInputs(new double[]{0, 1});
        System.out.println("0 XOR 1: " + (nn.getOutput()[0]));

        nn.setInputs(new double[]{1, 0});
        System.out.println("1 XOR 0: " + (nn.getOutput()[0]));

        nn.setInputs(new double[]{1, 1});
        System.out.println("1 XOR 1: " + (nn.getOutput()[0]) + "\n");
	}
	
	static void saveTrainingErrors(ArrayList<Double> errors) {
		
	    File file = new File("TrainingErrors.HS");
		try {
        	FileOutputStream is = new FileOutputStream(file);
        	OutputStreamWriter osw = new OutputStreamWriter(is);    
        	
        	Writer w = new BufferedWriter(osw);

      		for (int j=0;j < errors.size();j++) {
    			w.write( Double.toString(errors.get(j))+ " ");	
    		}

        	w.close();		
		} catch (IOException e) {
            System.err.println("Problem writing to the file statsTest.txt");
        }
	}
}
