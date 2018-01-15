package activations;

public class BipolarSigmoidActivation implements Activation {

	@Override
	public double activate(double weightedSum) {
		return ( 2/( 1 + Math.pow(Math.E,(-1*weightedSum)) ) - 1 );
	}

	@Override
	public double derivative(double outputOfActivate) {
		return 0.5 * (1 + outputOfActivate) * (1 - outputOfActivate);
	}

	@Override
	public Activation copy() {
		return new BipolarSigmoidActivation();
	}

}
