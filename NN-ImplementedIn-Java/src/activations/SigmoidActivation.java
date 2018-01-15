package activations;

public class SigmoidActivation implements Activation {

    public double activate(double weightedSum) {
        return 1.0 / (1 + Math.exp(-1.0 * weightedSum));
    }

    public double derivative(double outputOfActivate) {
        return outputOfActivate * (1.0 - outputOfActivate);
    }

    public Activation copy() {
        return new SigmoidActivation();
    }
}
