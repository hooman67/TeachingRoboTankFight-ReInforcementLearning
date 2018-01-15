package implementations;

import java.io.File;
import java.io.IOException;

import interfaces.NeuralNetInterface;

public class FinalNN implements NeuralNetInterface{
	
	 int numOfInputs;
	    int numOfHiddens;
	    double learningRate;
	    double momentumTerm;
	    
	    /* Weights */
	    double wights[][];
	    double nextWeights[][];
	    double prevWeights[][];
	    double changeInWeights[][];
	    
	    double secondLayerWeights[];
	    double nextSecondLayerWeights[];
	    double prevSecondLayerWeights[];
	    double changeInSecondLayerWeights[];
	    
	    
	    //hs lastDelta
	    double [ ]weightChangeHiddenToOutput;   
	    double [ ][ ]weightChangeInputToHidden;
	   
	    //hs deltaInputSum
	    double [ ] deltaInputSum_hidden;
	    //hs infoTerm
	    double [ ] errorInfoTerm_hidden;
	    
	    
	    double [ ]hiddenNueron;
	    double [ ]sigmoidForEachHiddenNueron;
	    
	    double outputNueron;
	    double outputSigmoid;
	    

	    
	    String outputType;
	        
	    /**
	    * Constructor. (Cannot be declared in an interface, but your implementation will need one)
	    * @param numOfInputs The number of inputs in your input vector
	    * @param numOfHiddens The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
	    * @param learningRate The learning rate coefficient
	    * @param momentumTerm The momentum coefficient
	    * @param outputType switch for the output type either be bipolar or binary. */
	    
	    FinalNN ( int numOfInputs,
	                int numOfHiddens,
	                double learningRate,
	                double momentumTerm,
	                String outputType )
	    {
	        this.numOfInputs = numOfInputs;
	        this.numOfHiddens = numOfHiddens;
	        this.learningRate = learningRate;
	        this.momentumTerm = momentumTerm;
	        
	        /* plus one below for the bias */
	        wights       = new double[ numOfInputs + 1 ][ numOfHiddens ];
	        nextWeights   = new double[ numOfInputs + 1 ][ numOfHiddens ];
	        prevWeights   = new double[ numOfInputs + 1 ][ numOfHiddens ];
	        weightChangeInputToHidden = new double[ numOfInputs + 1 ][ numOfHiddens ];
	        
	        /* plus one below for the bias */
	        secondLayerWeights = new double[ numOfHiddens + 1 ];
	        
	        hiddenNueron = new double[ numOfHiddens ];
	        sigmoidForEachHiddenNueron = new double[ numOfHiddens ];
	        
	        outputNueron = 0;
	        outputSigmoid = 0;
	        
	        changeInSecondLayerWeights = new double [ numOfHiddens + 1 ];
	        deltaInputSum_hidden = new double [ numOfHiddens ];
	        errorInfoTerm_hidden = new double [ numOfHiddens ];
	        
	        changeInWeights = new double[ numOfInputs + 1 ][ numOfHiddens ];
	        
	        /* Keeping the previous weight change for the momentum */       
	        weightChangeHiddenToOutput = new double [ numOfHiddens + 1 ];  
	        prevSecondLayerWeights = new double [numOfHiddens + 1];
	        nextSecondLayerWeights = new double [numOfHiddens + 1];
	        
	        this.outputType = outputType;       
	    }
	    

	    
	    /**
	     * This function calculates forward propagation for a 2-4-1 network
	     * @param X The input vector. An array of doubles.
	     * @return The value returned by the LookUpTable (LUT) or NN for this input vector
	     */
	    @Override
	    public double outputFor( double[ ] X ) 
	    {
	        int idx = 0;
	        int i = 0;
	        
	        /* reset the single output neuron to zero */
	        outputNueron = 0;
	        
	        /* reset all hidden neurons to zero from past. */
	        for ( idx = 0 ; idx < numOfHiddens ; idx++)
	        {
	            hiddenNueron[ idx ] = 0;
	        }        
	        
	        /* Calculate weighted sum for hidden neurons */
	        for ( idx = 0; idx < numOfHiddens; idx++)
	        {
	            /* add the bias weighted sum for each hidden neuron first */
	            hiddenNueron[idx] = bias * wights[0][idx];
	            
	            /* Add weighted sum for each hidden neuron based on the inputs */
	            for ( i = 0; i < numOfInputs; i++)
	            {
	                hiddenNueron[ idx ] += X[i] * wights[ i + 1 ][ idx ];
	            }
	            
	            /* For the binary inputs use binary sigmoid on all neurons. */
	            if( "binary" == outputType ) 
	            {
	                sigmoidForEachHiddenNueron[ idx ] = customSigmoid( hiddenNueron[ idx ] );                
	            }
	            /* For the bipolar case, use bipolar sigmoid on all neurons. */
	            else if( "bipolar" == outputType )
	            {
	                sigmoidForEachHiddenNueron[ idx ] = sigmoid( hiddenNueron[ idx ] );                
	            }
	            else
	            {
	                System.out.println( "Error, cannot decide the type of the sigmoid" );
	                /* return some arbitrary negative number due to error */
	                return( -100 );
	            }            
	        }
	        
	        /* final stage here for the forward propagation calculation
	         * Calculate the output based on the calculate hidden neurons */
	        outputNueron = bias * secondLayerWeights[0];
	        for ( idx = 0; idx < numOfHiddens; idx++)
	        {
	             outputNueron += sigmoidForEachHiddenNueron[ idx ] * secondLayerWeights[ idx + 1 ];
//	            outputNueron += hiddenNueron[ idx ] * weightHiddenToOutput[ idx + 1 ];
	        }
	        
	        /* Finally here for the forward propagation
	         * Calculate obtained output value based on the sigmoid function */
	        if( "binary" == outputType ) 
	        {            
	            outputSigmoid = customSigmoid( outputNueron );
	        }
	        else
	        {
	            outputSigmoid = sigmoid( outputNueron );            
	        }
	        
	        return( outputSigmoid );
	    }

	    /**
	     * This method will tell the NN or the LUT the output
	     * value that should be mapped to the given input vector. I.e.
	     * the desired correct output value for an input.
	     * This method is backward propagation method.
	     * @param X The input vector
	     * @param expectedOutputValue The new value to learn
	     * @return The error in the output for that input vector
	     */
	    @Override
	    public double train( double[] X, double expectedOutputValue ) 
	    {
	        /* delta (error) of the output */
	        double errorInfoTerm_output;
	        int idx;
	        int input;
	        
	        /* start from the output neuron, calculate errors and delta weights
	         * going backwards */
	        
	        /* compute the error information term, delta. */
	        if( outputType.equals( "binary" ) )
	        {
	            errorInfoTerm_output = 
	                    ( expectedOutputValue - outputSigmoid ) *
	                    ( outputSigmoid * ( 1 - outputSigmoid ) );  
	        }
	        else if( outputType.equals( "bipolar" ) )           
	        {
	            errorInfoTerm_output = 
	                    ( expectedOutputValue - outputSigmoid) *
	                    0.5 * ( 1 + outputSigmoid ) * ( 1 - outputSigmoid );
	        }
	        else
	        {
	            System.out.println( "Error, cannot decide the type of the sigmoid" );
	            return( -100 );            
	        }
	        
	        /* calculate bias correction term (step_size*delta*u) for the bias clamped to 1 
	         * that goes to the output */
	        changeInSecondLayerWeights[ 0 ] = learningRate * errorInfoTerm_output * bias;
	        /* calculate weight correction term for hidden to output edges */
	        for( idx = 1; idx < numOfHiddens + 1; idx++ )
	        {
	            changeInSecondLayerWeights[ idx ] = learningRate *
	                                                errorInfoTerm_output *
	                                                sigmoidForEachHiddenNueron[ idx - 1  ];
	        }

	        /* ---------------------------------------------------- */
	        /* for each hidden neuron calculate the delta input sum that
	         * is from the output (layer above)
	         * ie.: delta_out x weight_hi */
	        for( idx = 0; idx < numOfHiddens; idx++ )
	        {
	            /* todo weightHiddenToOutput[ 0 ] is the bias to output
	               discuss how to handle that. */
	            /* (f' * outputError * weighthiddentooutput) */
	            
	            deltaInputSum_hidden[ idx ] =
	                    errorInfoTerm_output * secondLayerWeights[ idx + 1 ];        
	       
	            /* Now ready to calculate the "delta" (error) for each neuron in hidden layer */
	            if( outputType.equals( "binary" ) )
	            {
	                errorInfoTerm_hidden[ idx ] =
	                        deltaInputSum_hidden[ idx ] *
	                        ( ( sigmoidForEachHiddenNueron[ idx ] *
	                                ( 1 - sigmoidForEachHiddenNueron[ idx ] ) ) );
	            }
	            else
	            {
	                errorInfoTerm_hidden[ idx ] =
	                        deltaInputSum_hidden[idx] *
	                        ( 0.5 * ( 1 + sigmoidForEachHiddenNueron[idx] ) *
	                                ( 1 - sigmoidForEachHiddenNueron[idx] ) );
	            }
	        }

	        /* Compute weight correction term for input to hidden edges */
	        for ( idx = 0; idx < numOfHiddens; idx++ )
	        {
	            /* Bias correction term */
	            changeInWeights[ 0 ][ idx ] =
	                                               learningRate *
	                                               errorInfoTerm_hidden[ idx ] * bias;
	           for ( input = 1; input < numOfInputs+1; input++ )
	           {
	               changeInWeights[ input ][ idx ]=
	                                                     learningRate *
	                                                     errorInfoTerm_hidden[ idx ] *
	                                                     X[ input-1 ];
	           }
	       }
	        
	        /* ---------------------------------------------------- */
	        /* Update the weights
	         * Now I am ready to compute all the new weights
	         * ie.: W_new = W_old +
	         *             (momentum x previous_weight_change) +
	         *             ( rho x delta x input ) 
	         */
	        /* ---------------------------------------------------- */         
	        for ( idx = 0; idx < numOfHiddens + 1; idx++ )
	        {
	            /* update the weights between the hidden layer and the output */
	            nextSecondLayerWeights[ idx ] =
	                    secondLayerWeights[ idx ] +
	                    ( momentumTerm * weightChangeHiddenToOutput[ idx ] ) +
	                    changeInSecondLayerWeights[ idx ];
	            
	            /* update the weight change keeper */            
	            weightChangeHiddenToOutput[ idx ] = secondLayerWeights[ idx ] - 
	                                                prevSecondLayerWeights [ idx ];                    
	            
	            /* update and keep the last and current weights */
	            prevSecondLayerWeights [ idx ] = secondLayerWeights[ idx ];
	            secondLayerWeights[ idx ] = nextSecondLayerWeights[ idx ];
	        }
	        
	        /* Update input to hidden weights */        
	        for( idx = 0; idx < numOfHiddens; idx++ )
	        {
	            for ( input = 0; input < numOfInputs+1; input++ )
	            {
	               nextWeights[ input ][ idx ] =
	                       wights[ input ][ idx ] +
	                       ( momentumTerm * weightChangeInputToHidden[ input ][ idx ] ) +
	                       changeInWeights[ input ][ idx ];
	               
	               /* update the weight change keeper */
	               weightChangeInputToHidden[ input ][ idx ] = wights[ input ][ idx ] -
	                                                           prevWeights[ input ][ idx ];
	               
	               /* update and keep the last and current weights */
	               prevWeights[ input ][ idx ] = wights[ input ][ idx ];
	               wights[ input ][ idx ] = nextWeights[ input ][ idx ];               
	            }            
	        }              
	        
	        /* return the difference between received output value 
	         * and the expected output value */
	        return ( expectedOutputValue - outputSigmoid );
	    }

	@Override
	public void save(File argFile) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void load(String argFileName) throws IOException {
		// TODO Auto-generated method stub
		
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
        /* Randomly initialize the weights between the inputs and
         * the hidden layer
         * +1 below for the bias neuron */
        for( int i = 0; i < numOfHiddens + 1; i++ )
        {
            secondLayerWeights[ i ] = (Math.random() * 1) - 0.5;
            prevSecondLayerWeights [ i ]   = 0;
            nextSecondLayerWeights [ i ]   = 0;
            weightChangeHiddenToOutput [ i ] = 0;
        }            
        
        /* Randomly initialize the weights on edges between hidden layer
         * and the output
         *  +1 below for the bias neuron */
        for(  int i = 0; i < numOfInputs + 1; i++ )    
        {
            /* +1 below for the bias neuron */
            for( int j = 0; j < numOfHiddens; j++ )       
            {
                wights[ i ][ j ] = (Math.random() * 1) - 0.5;
                nextWeights[ i ][ j ]   = 0;
                prevWeights[ i ][ j ]   = 0;  
                weightChangeInputToHidden[ i ][ j ] = 0;
            }
        }
	}

	@Override
	public void zeroWeights() {
        /* Randomly initialize the weights between the inputs and
         * the hidden layer
         * +1 below for the bias neuron */
        for( int i = 0; i < numOfHiddens + 1; i++ )
        {
            secondLayerWeights[ i ] = 0;
            prevSecondLayerWeights [ i ]   = 0;
            nextSecondLayerWeights [ i ]   = 0;
            weightChangeHiddenToOutput [ i ] = 0;
        }            
        
        /* Randomly initialize the weights on edges between hidden layer
         * and the output
         *  +1 below for the bias neuron */
        for(  int i = 0; i < numOfInputs + 1; i++ )    
        {
            /* +1 below for the bias neuron */
            for( int j = 0; j < numOfHiddens; j++ )       
            {
                wights[ i ][ j ] = 0;
                nextWeights[ i ][ j ]   = 0;
                prevWeights[ i ][ j ]   = 0;  
                weightChangeInputToHidden[ i ][ j ] = 0;
            }
        }
	}

}
