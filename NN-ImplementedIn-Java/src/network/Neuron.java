package network;

import java.util.ArrayList;
import java.util.List;

import activations.Activation;

public class Neuron {

    private List<Synapse> inputs;
    private Activation activation;
    private double output;
    private double derivative;
    private double weightedSum;
    private double error;

    public Neuron(Activation activationStrategy) {
        inputs = new ArrayList<Synapse>();
        this.activation = activationStrategy;
        error = 0;
    }
    
    public Neuron(Activation activationStrategy, double output) {
        inputs = new ArrayList<Synapse>();
        setOutput(output);
        this.activation = activationStrategy;
        error = 0;
    }

    public void addInput(Synapse input) {
        inputs.add(input);
    }

    public List<Synapse> getInputs() {
        return this.inputs;
    }

    public double[] getWeights() {
        double[] weights = new double[inputs.size()];

        int i = 0;
        for(Synapse synapse : inputs) {
            weights[i] = synapse.getWeight();
            i++;
        }

        return weights;
    }

    private void calculateWeightedSum() {
        weightedSum = 0;
        for(Synapse synapse : inputs) {
            weightedSum += synapse.getWeight() * synapse.getSourceNeuron().getOutput();
        }
    }

    public void activate() {
        calculateWeightedSum();
        output = activation.activate(weightedSum);
        derivative = activation.derivative(output);
    }

    public double getOutput() {
        return this.output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getDerivative() {
        return this.derivative;
    }

    public Activation getActivationStrategy() {
        return activation;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }
}
