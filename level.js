class Level {
    constructor(inputCount, outputCount) {
        this.weights = tf.randomNormal([inputCount, outputCount]); // Initialize weights randomly
        this.biases = tf.randomNormal([outputCount]); // Initialize biases randomly
    }

    static sigmoid(x) {
        return tf.div(tf.scalar(1), tf.add(tf.scalar(1), tf.exp(tf.neg(x))));
    }

    static feedForward(givenInputs, level) {
        const inputs = tf.tensor2d([givenInputs]); // Convert input array to a tensor

        const sum = tf.matMul(inputs, level.weights).add(level.biases); // Perform matrix multiplication and add biases
        const outputs = Level.sigmoid(sum); // Apply sigmoid activation function

        const outputArray = outputs.arraySync()[0].map(value => (value >= 0.5 ? 1 : 0));

        return { input: givenInputs, output: outputArray };
    }
}