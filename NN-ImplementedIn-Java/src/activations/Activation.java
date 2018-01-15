package activations;

public interface Activation {

    double activate(double weightedSum);
    double derivative(double outputOfActivate);
    Activation copy();
}
