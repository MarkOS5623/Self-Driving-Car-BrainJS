class Car{
    constructor(x, y, width, height, controlType, maxSpeed = 4) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.speed = 0;
        this.acceleration = 0.2;
        this.maxSpeed = maxSpeed;
        this.friction = 0.05;
        this.angle = 0;
        this.damaged = false;
        this.useBrain = controlType == "AI";
        this.trained = false;
        this.trainingData = [];
        if (controlType !== "DUMMY") {
            this.sensor = new Sensor(this);
            const input = tf.input({shape: [5]});
            const dense1 = tf.layers.dense({ units: 6, activation: 'relu'}).apply(input);
            const dense2 = tf.layers.dense({ units: 4, activation: 'sigmoid'}).apply(dense1);
            this.network = tf.model({inputs: input, outputs: dense2});
        }
        this.controls = new Controls(controlType);
    }

    async Trainer() { // trains the networks
        try {
            if (!this.network) {
                console.error("No network");
                return;
            }
            if (this.trainingInProgress) {
                console.log("Training already in progress");
                return;
            }
            this.trainingInProgress = true;
            const storedTrainingData = JSON.parse(localStorage.getItem('carTrainingData'));
            if (!storedTrainingData || storedTrainingData.length === 0) {
                const randomInput = generateRandomInput();
                const randomOutput = generateRandomOutput(); 
                const xs = tf.tensor2d([randomInput]);
                const ys = tf.tensor2d([randomOutput]);
                const config = {
                    epochs: 5,  
                    shuffle: true,
                    validationSplit: 0.1, 
                    learningRate: 0.2,
                };
                await this.network.compile({ loss: 'meanSquaredError', optimizer: 'adam' });
                await this.network.fit(xs, ys, config);
                console.log("Model trained with random data");
            } else {
                const inputArrays = storedTrainingData.map(entry => entry.input);
                const processoredData = DataProcessor(inputArrays);
                for (let i = 0; i < processoredData.length; i++) 
                    this.trainingData.push({input: processoredData[i].input, output: processoredData[i].output})
                
                const xs = tf.tensor2d(this.trainingData.map(data => data.input));
                const ys = tf.tensor2d(this.trainingData.map(data => data.output));
                const config = {
                    epochs: 5,  // Adjust the number of epochs as needed
                    validationSplit: 0.1,  // Adjust validation split as needed
                    learningRate: 0.4,
                };
                // Compile and fit the model
                await this.network.compile({ loss: 'meanSquaredError', optimizer: 'adam' });
                await this.network.fit(xs, ys, config);
                console.log("Model trained with stored training data");
            }
            this.trained = true;
        } catch (error) {
            console.log("Error during training: ", error);
        } finally {
            this.trainingInProgress = false;
        }
    }
    
    update(roadBorders, traffic){
        if(!this.damaged){
            this.#move();
            this.polygon = this.#createPolygon();
            this.damaged = this.#assessDamage(roadBorders, traffic);
        }
        if (this.sensor) {
            this.sensor.update(roadBorders, traffic);
            const offsets = this.sensor.readings.map(s => s == null ? 0 : 1 - s.offset);

            // Make predictions with the neural network
            const inputTensor = tf.tensor2d([offsets]);
            const outputTensor = this.network.predict(inputTensor);
            const outputs = Array.from(outputTensor.dataSync()).map(value => (value >= 0.5 ? 1 : 0));
        
            // Dispose of the input and output tensors
            inputTensor.dispose();
            outputTensor.dispose();
        
            if (this.useBrain) {
                this.controls.forward = outputs[0];
                this.controls.left = outputs[1];
                this.controls.right = outputs[2];
                this.controls.reverse = 0; // reversing is not allowed encourges cars to move out of the way instead of "kiting" beheind cars
            }
            const trainingExample = { input: offsets, output: outputs };
            this.trainingData.push(trainingExample);
        }
    }

    #assessDamage(roadBorders,traffic){
        for(let i = 0; i < roadBorders.length; i++)
            if(polysIntersect(this.polygon,roadBorders[i]))
                return true;
        for(let i = 0; i < traffic.length; i++)
            if(polysIntersect(this.polygon,traffic[i].polygon))
                return true;
        return false;
    }

    #createPolygon(){
        const points = [];
        const rad = Math.hypot(this.width,this.height) / 2;
        const alpha = Math.atan2(this.width,this.height);
        points.push({
            x:this.x-Math.sin(this.angle - alpha) * rad,
            y:this.y-Math.cos(this.angle - alpha) * rad
        });
        points.push({
            x:this.x-Math.sin(this.angle + alpha) * rad,
            y:this.y-Math.cos(this.angle + alpha) * rad
        });
        points.push({
            x:this.x-Math.sin(Math.PI + this.angle - alpha) * rad,
            y:this.y-Math.cos(Math.PI + this.angle - alpha) * rad
        });
        points.push({
            x:this.x - Math.sin(Math.PI + this.angle + alpha) * rad,
            y:this.y - Math.cos(Math.PI + this.angle + alpha) * rad
        });
        return points;
    }

    #move(){
        if(this.controls.forward)
            this.speed += this.acceleration;
        if(this.controls.reverse)
            this.speed -= this.acceleration;
        if(this.speed > this.maxSpeed)
            this.speed = this.maxSpeed;
        if(this.speed <- this.maxSpeed / 2)
            this.speed = -this.maxSpeed / 2;
        if(this.speed > 0)
            this.speed -= this.friction;
        if(this.speed < 0)
            this.speed += this.friction;   
        if(Math.abs(this.speed) < this.friction)
            this.speed = 0;
        if(this.speed != 0){
            const flip = this.speed > 0 ? 1 :- 1;
            if(this.controls.left)
                this.angle += 0.03 * flip;
            if(this.controls.right)
                this.angle -= 0.03 * flip;
        }
        this.x -= Math.sin(this.angle) * this.speed;
        this.y -= Math.cos(this.angle) * this.speed;
    }

    draw(ctx, color){
        if(this.damaged)
            ctx.fillStyle = "gray";
        else
            ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(this.polygon[0].x, this.polygon[0].y);
        for(let i = 1; i < this.polygon.length; i++)
            ctx.lineTo(this.polygon[i].x, this.polygon[i].y);
        ctx.fill();
        if(this.sensor)
            this.sensor.draw(ctx);
    }
}