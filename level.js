class Level { // simulates level from a neural network
    constructor(inputCount, outputCount) {
        this.inputs = new Array(inputCount);
        this.outputs = new Array(outputCount);
        this.biases = new Array(outputCount);

        this.weights = [];
        for (let i = 0; i < inputCount; i++) { // each input gets a weight array in the size of the output
            this.weights[i] = new Array(outputCount); // each input has weights for each output
        }

        Level.#randomize(this);
    }

    static #randomize(level) {
        for (let i = 0; i < level.inputs.length; i++) {
            for (let j = 0; j < level.outputs.length; j++) {
                level.weights[i][j] = Math.random() * 2 - 1; // give random weights to each of output array values
            }
        }
        for (let i = 0; i < level.biases.length; i++) {
            level.biases[i] = Math.random() * 2 - 1; // gives random biases for each output
        }
    }

    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    static feedForward(givenInputs, level) {

        for (let i = 0; i < level.inputs.length; i++) { // inserts the sensor reading into the level
            level.inputs[i] = givenInputs[i];
        }

        const output = [];

        for (let i = 0; i < level.outputs.length; i++) {
            let sum = 0;
            for (let j = 0; j < level.inputs.length; j++) { // sums the multiplication of the input value times the weights of each node
                sum += level.inputs[j] * level.weights[j][i];
            }

            level.outputs[i] = Level.sigmoid(sum - level.biases[i]); 
            const roundedOutput = level.outputs[i] >= 0.5 ? 1 : 0; // rounds the results of output into 1's and 0's
            output.push(roundedOutput);
        }

        return { input: givenInputs, output: output }; // Return the output in the training data format
    }
}

