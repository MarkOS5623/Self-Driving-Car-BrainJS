function lerp(A, B, t){ // returns a number between two numbers used mainly to manipulate baises and weights
    return A + (B - A) * t;
}

function getIntersection(A,B,C,D){ 
    const tTop=(D.x-C.x)*(A.y-C.y)-(D.y-C.y)*(A.x-C.x);
    const uTop=(C.y-A.y)*(A.x-B.x)-(C.x-A.x)*(A.y-B.y);
    const bottom=(D.y-C.y)*(B.x-A.x)-(D.x-C.x)*(B.y-A.y);
    if(bottom!=0){
        const t=tTop/bottom;
        const u=uTop/bottom;
        if(t >= 0 && t <= 1 && u >= 0 && u <= 1){
            return {
                x:lerp(A.x,B.x,t),
                y:lerp(A.y,B.y,t),
                offset:t
            }
        }
    }
    return null;
}

function polysIntersect(poly1, poly2){ // checks if two different polygons cross each other
    for(let i = 0; i < poly1.length; i++){
        for(let j = 0; j < poly2.length; j++){
            const touch = getIntersection(
                poly1[i],
                poly1[(i + 1) % poly1.length],
                poly2[j],
                poly2[(j + 1) % poly2.length]
            );
            if(touch)
                return true;
        }
    }
    return false;
}

function getRGBA(value){
    const alpha = Math.abs(value);
    const R = value < 0 ? 0 : 255;
    const G = R;
    const B = value > 0 ? 0 : 255;
    return "rgba("+ R + "," + G + "," + B + "," + alpha + ")";
}
               
async function save() {
    try {
        if (bestCar && bestCar.network) {
            const saveResult = await bestCar.network.save('localstorage://bestModel');
            localStorage.setItem('carTrainingData', JSON.stringify(bestCar.trainingData));
            console.log("Save successful");
        } else {
            console.error("No best car or network found to save.");
        }
    } catch (error) {
        console.error("Error saving brain and training data:", error);
    }
}

async function saveTrained() { // saves a pre trained model just incase
    try {
        if (bestCar && bestCar.network) {
            const PreResult = await bestCar.network.save('localstorage://pretrainedModel');
            localStorage.setItem('carTrainingData', JSON.stringify(bestCar.trainingData));
            console.log("Save successful");
        } else {
            console.error("No best car or network found to save.");
        }
    } catch (error) {
        console.error("Error saving brain and training data:", error);
    }
}

function discard() {
    try {
        localStorage.removeItem("carTrainingData");
        console.log("Delete successful");
    } catch (error) {
        console.error("Error deleting brain and training data:", error);
    }
}

async function loadTrained() {
    try {
        const model = await tf.loadLayersModel('localstorage://pretrainedModel');
        const PreResult1 = await model.save('localstorage://pretrainedModel');
        const PreResult2 = await model.save('localstorage://bestModel');
        console.log("Upload successful");
    } catch (error) {
        console.error("Error uploading brain and training data:", error);
    }
}

function generateRandomInput() {// return an arrya of 5 random numbers between -1 and 1
    return [
        Math.random() * 2 - 1, 
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
        Math.random() * 2 - 1
    ];
}

function generateRandomOutput() { // return an arrya of 4 random booleans(ones or zeros)
    return Array.from({ length: 4 }, () => Math.round(Math.random()));
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function DataProcessor(givenInputs) { // proccess the training data to get varied data
    let output = [];
    const final = [];
    for (let i = 0; i < givenInputs.length; i++) { // for each array in given inputs
        let sum = 0;
        for (let j = 0; j < givenInputs[i].length; j++) { // multiple the input value 5 times(number of outputs) with a random value between -1 and 1
            for(let k = 0; k < 5; k++){
                sum += givenInputs[i][j] * (Math.random() * 2 - 1);
            }
            const out = sigmoid(sum - (Math.random() * 2 - 1)); // use sigmoid to get a value between 0 and 1 ie valid weight
            const roundedOutput = out >= 0.5 ? 1 : 0; // round the result into booleans for the car controls
            output.push(roundedOutput);
            if(output.length == 4){
                final.push(output);
                output = [];
            }
        }
        if(final.length == givenInputs.length) break; // makes sure the number of inputs and output arrays are the same
    }
    const processoredData = [];
    for (let i = 0; i < givenInputs.length; i++) { // puts the data in the currect format for trainign using tensors
        processoredData.push({input: givenInputs[i], output: final[i]})
    }
    return processoredData;
}

async function mutateModel(model, mutationRate) { // function for mutating ie alter the weights and biases of a nerual network
    const mutatedModel = await tf.tidy(() => {
        const layers = model.layers.map(layer => { // each layer has two weight arrays one contains the weights and the other the biases
            const layerConfig = layer.getConfig();
            const layerClass = tf.layers[layerConfig.className];
            if (layerClass) {
                const mutatedLayer = layerClass(layerConfig);  // Create a new layer with the same configuration
                if (layer.trainable) {
                    // Mutates weights
                    const weights = layer.getWeights();
                    const mutatedWeights = weights.map(weight => {
                        const shape = weight.shape;
                        const values = weight.dataSync().slice();
                        const mutatedValues = values.map(val => {
                            if (Math.random() < mutationRate) 
                                return val + (Math.random() - 0.5);
                            return val;
                        });
                        return tf.tensor(mutatedValues, shape);
                    });
                    mutatedLayer.setWeights(mutatedWeights);
                    // Mutatse biases
                    const biases = layer.getWeights()[1]; 
                    const biasesValues = biases.dataSync().slice();
                    const mutatedBiases = biasesValues.map(val => {
                        if (Math.random() < mutationRate) 
                            return val + (Math.random() - 0.5);
                        return val;
                    });
                    mutatedLayer.getWeights()[1].dispose();  // Dispose old biases
                    mutatedLayer.setWeights([mutatedWeights[0], tf.tensor(mutatedBiases, biases.shape)]);
                }
                return mutatedLayer;
            }
            return null;
        }).filter(layer => layer !== null);
        return tf.model({ inputs: model.inputs, outputs: model.outputs, layers: layers });
    });
    return mutatedModel;
}