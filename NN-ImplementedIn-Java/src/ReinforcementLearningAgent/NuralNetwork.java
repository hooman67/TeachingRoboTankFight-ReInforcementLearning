package nnClassAssignment;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import activations.Activation;
import activations.BipolarSigmoidActivation;
import activations.SigmoidActivation;
import network.Layer;
import network.Neuron;
import network.Synapse;

public class NuralNetwork implements NeuralNetInterface {
	
    private String name = "HsTrained XOR Network";
    double argLearningRate;
    double argMomentumTerm;
	double currentEpoch;
    
    private List<Layer> layers;
    private Layer input;
    private Layer output;


	public NuralNetwork(
			int argNumInputs,
			int argNumOutputs,
			int argNumHidden,
			Activation activation,
			double argLearningRate,
			double argMomentumTerm) {

        layers = new ArrayList<Layer>();
		this.argLearningRate = argLearningRate;
		this.argMomentumTerm = argMomentumTerm;

		
		//specify input layer
        Layer inputLayer = new Layer( null, new Neuron(activation, 1) );
       
        for(int i=0; i< argNumInputs; i++){
            inputLayer.addNeuron( new Neuron(activation, 0) );
        }
        
        
        //specify hidden layer
        Layer hiddenLayer = new Layer(
        		inputLayer,
        		new Neuron(activation, 1) );
        
        for(int i=0; i< argNumHidden; i++){
        	hiddenLayer.addNeuron( new Neuron(activation) );
        }
        
        
        //specify output layer
        Layer outputLayer = new Layer(hiddenLayer);
        for(int i=0; i< argNumOutputs; i++){
        	outputLayer.addNeuron( new Neuron(activation) );
        }


        addLayer(inputLayer);
        addLayer(hiddenLayer);
        addLayer(outputLayer);
	}
	
    void addLayer(Layer layer) {
        layers.add(layer);

        if(layers.size() == 1) {
            input = layer;
        }

        if(layers.size() > 1) {
            //clear the output flag on the previous output layer, but only if we have more than 1 layer
            Layer previousLayer = layers.get(layers.size() - 2);
            previousLayer.setNextLayer(layer);
        }

        output = layers.get(layers.size() - 1);
    }

    public void setInputs(double[] inputs) {
        if(input != null) {

            int biasCount = input.hasBias() ? 1 : 0;

            if(input.getNeurons().size() - biasCount != inputs.length) {
                throw new IllegalArgumentException("The number of inputs must equal the number of neurons in the input layer");
            }

            else {
                List<Neuron> neurons = input.getNeurons();
                for(int i = biasCount; i < neurons.size(); i++) {
                    neurons.get(i).setOutput(inputs[i - biasCount]);
                }
            }
        }
    }

    public String getName() {
        return name;
    }

    public double[] getOutput() {

        double[] outputs = new double[output.getNeurons().size()];

        for(int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.feedForward();
        }

        int i = 0;
        for(Neuron neuron : output.getNeurons()) {
            outputs[i] = neuron.getOutput();
            i++;
        }

        return outputs;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public double[] getWeights() {

        List<Double> weights = new ArrayList<Double>();

        for(Layer layer : layers) {

            for(Neuron neuron : layer.getNeurons()) {

                for(Synapse synapse: neuron.getInputs()) {
                    weights.add(synapse.getWeight());
                }
            }
        }

        double[] allWeights = new double[weights.size()];

        int i = 0;
        for(Double weight : weights) {
            allWeights[i] = weight;
            i++;
        }

        return allWeights;
    }

    public void setWeights(ArrayList<Double> weights) {
        if(weights == null) {
            throw new IllegalArgumentException("Input weights from file empty");
        }

        Iterator<Double> it = weights.iterator();
        for(Layer layer : layers) {

            for(Neuron neuron : layer.getNeurons()) {

                for(Synapse synapse: neuron.getInputs()) {
                    synapse.setWeight(it.next());
                }
            }
        }
    }

    public void hsSave() {
    	 
    	String fileName = name.replaceAll(" ", "") + "-" + new Date().getTime() +".HS";
    	File file = new File(fileName);
    	save(file);
    }
    
    public ArrayList<Double> train(double[][] inputs, double[][] expectedOutputs, double errorThreshold) {

        double error = 1000;
        
        int epoch = 1;

        ArrayList<Double> out = new ArrayList<>();

        while(error > errorThreshold) {
        	
            error = backpropagate(inputs, expectedOutputs);

            if(epoch % 100 == 0)
            	out.add(error);
            if( epoch % 500 == 0 ){
            	System.out.println("Error for epoch " + epoch + ": " + error);            	
            }
            
            epoch++;
            currentEpoch = epoch;
        }
        
        System.out.println("Final epoch: " +epoch +";  finalError: " + error);
        return out;
    }

    public double backpropagate(double[][] inputs, double[][] expectedOutputs) {

        double error = 0;

        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<Synapse, Double>();

        for (int i = 0; i < inputs.length; i++) {

            double[] input = inputs[i];
            double[] expectedOutput = expectedOutputs[i];

            List<Layer> layers = getLayers();

            setInputs(input);
            double[] output = getOutput();

            //First step of the backpropagation algorithm. Backpropagate errors from the output layer all the way up
            //to the first hidden layer
            for (int j = layers.size() - 1; j > 0; j--) {
                Layer layer = layers.get(j);

                for (int k = 0; k < layer.getNeurons().size(); k++) {
                    Neuron neuron = layer.getNeurons().get(k);
                    double neuronError = 0;

                    if (layer.isOutputLayer()) {
                        //the order of output and expected determines the sign of the delta. if we have output - expected, we subtract the delta
                        //if we have expected - output we add the delta.
                        neuronError = neuron.getDerivative() * (output[k] - expectedOutput[k]);
                    } else {
                        neuronError = neuron.getDerivative();

                        double sum = 0;
                        List<Neuron> downstreamNeurons = layer.getNextLayer().getNeurons();
                        for (Neuron downstreamNeuron : downstreamNeurons) {

                            int l = 0;
                            boolean found = false;
                            while (l < downstreamNeuron.getInputs().size() && !found) {
                                Synapse synapse = downstreamNeuron.getInputs().get(l);

                                if (synapse.getSourceNeuron() == neuron) {
                                    sum += (synapse.getWeight() * downstreamNeuron.getError());
                                    found = true;
                                }

                                l++;
                            }
                        }

                        neuronError *= sum;
                    }

                    neuron.setError(neuronError);
                }
            }

            //Second step of the backpropagation algorithm. Using the errors calculated above, update the weights of the
            //network
            for(int j = layers.size() - 1; j > 0; j--) {
                Layer layer = layers.get(j);

                for(Neuron neuron : layer.getNeurons()) {

                    for(Synapse synapse : neuron.getInputs()) {

                        double delta = argLearningRate * neuron.getError() * synapse.getSourceNeuron().getOutput();

                        if(synapseNeuronDeltaMap.get(synapse) != null) {
                            double previousDelta = synapseNeuronDeltaMap.get(synapse);
                            delta += argMomentumTerm * previousDelta;
                        }

                        synapseNeuronDeltaMap.put(synapse, delta);
                        synapse.setWeight(synapse.getWeight() - delta);
                    }
                }
            }

            output = getOutput();
            error += error(output, expectedOutput);
        }

        return error;
    }

    public double error(double[] actual, double[] expected) {

        if (actual.length != expected.length) {
            throw new IllegalArgumentException("The lengths of the actual and expected value arrays must be equal");
        }

        double sum = 0;

        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i] - actual[i], 2);
        }

        return sum / 2;
    }
    
    
	//Required by the interface only
	@Override
	public double outputFor(double[] X) {
		return getOutput()[0];
	}

	@Override
	public double train(double[] X, double argValue) {
 
        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<Synapse, Double>();

        List<Layer> layers = getLayers();

        setInputs(X);
        double[] output = getOutput();

        //First step of the backpropagation algorithm. Backpropagate errors from the output layer all the way up
        //to the first hidden layer
        for (int j = layers.size() - 1; j > 0; j--) {
        	Layer layer = layers.get(j);

        	for (int k = 0; k < layer.getNeurons().size(); k++) {
        		Neuron neuron = layer.getNeurons().get(k);
        		double neuronError = 0;

        		if (layer.isOutputLayer()) {
        			//the order of output and expected determines the sign of the delta. if we have output - expected, we subtract the delta
        			//if we have expected - output we add the delta.
        			neuronError = neuron.getDerivative() * (output[0] - argValue);
        		} else {
        			neuronError = neuron.getDerivative();

        			double sum = 0;
        			List<Neuron> downstreamNeurons = layer.getNextLayer().getNeurons();
        			for (Neuron downstreamNeuron : downstreamNeurons) {

        				int l = 0;
        				boolean found = false;
        				while (l < downstreamNeuron.getInputs().size() && !found) {
        					Synapse synapse = downstreamNeuron.getInputs().get(l);

        					if (synapse.getSourceNeuron() == neuron) {
        						sum += (synapse.getWeight() * downstreamNeuron.getError());
        						found = true;
        					}

        					l++;
        				}
        			}

        			neuronError *= sum;
        		}

        		neuron.setError(neuronError);
        	}
        }

        //Second step of the backpropagation algorithm. Using the errors calculated above, update the weights of the
        //network
        for(int j = layers.size() - 1; j > 0; j--) {
        	Layer layer = layers.get(j);

        	for(Neuron neuron : layer.getNeurons()) {

        		for(Synapse synapse : neuron.getInputs()) {

        			double delta = argLearningRate * neuron.getError() * synapse.getSourceNeuron().getOutput();

        			if(synapseNeuronDeltaMap.get(synapse) != null) {
        				double previousDelta = synapseNeuronDeltaMap.get(synapse);
        				delta += argMomentumTerm * previousDelta;
        			}

        			synapseNeuronDeltaMap.put(synapse, delta);
        			synapse.setWeight(synapse.getWeight() - delta);
        		}
        	}
        }

        output = getOutput();
        return Math.pow(output[0] - argValue, 2) / 2;  
	}

	@Override
	public void save(File argFile) {
		
		try {
        	FileOutputStream is = new FileOutputStream(argFile);
        	OutputStreamWriter osw = new OutputStreamWriter(is);    
        	
        	Writer w = new BufferedWriter(osw);
        	
        	double weights[] = getWeights();
        	
      		for (int j=0;j < weights.length;j++) {
    			w.write( Double.toString(weights[j])+ " ");	
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
		
		ArrayList<Double> weights = new ArrayList<>();
		
		try{
			Scanner scanner =  new Scanner(fFilePath,ENCODING.name());
			scanner.useDelimiter(" ");
			
			while ( scanner.hasNext() ) {
				weights.add(scanner.nextDouble());
			}
		} catch(IOException e) {System.out.println("could not open file to load");}
		
		setWeights(weights);
	}

	@Override
	public double sigmoid(double x) {
		return ( 2/( 1 + Math.pow(Math.E,(-1*x)) ) - 1 );
	}

	@Override
	public double customSigmoid(double x) {
		return 1.0 / (1 + Math.exp(-1.0 * x));
	}

	@Override
	public void initializeWeights() {
		
        for(Layer layer : layers) {
            for(Neuron neuron : layer.getNeurons()) {
                for(Synapse synapse : neuron.getInputs()) {
                    synapse.setWeight((Math.random() * 1) - 0.5);
                }
            }
        }
	}

	@Override
	public void zeroWeights() {
		
        for(Layer layer : layers) {
            for(Neuron neuron : layer.getNeurons()) {
                for(Synapse synapse : neuron.getInputs()) {
                    synapse.setWeight(0);
                }
            }
        }
	}
	
}
