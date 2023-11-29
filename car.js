
class Car{
    constructor(x , y, width, height, controlType, maxSpeed = 4, data, mutate = false){
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
        if(controlType != "DUMMY"){
            this.sensor = new Sensor(this);
            if(data){
                this.network = new brain.NeuralNetwork();
                this.network.fromJSON(data);
                if(mutate != true){
                    this.trained = true;
                } else {
                    this.trained = false;
                }
            } else{
                this.network = new brain.NeuralNetwork({
                    inputSize: 5,
                    hiddenLayers: [6],
                    outputSize: 4,
                    learningRate: 0.5,
                    iterations: 5000,
                });
            }
        }
        this.controls = new Controls(controlType);
    }
    
    Trainer() {
        // Get input from sensor readings (distances to other objects)
        const offsets = this.sensor.readings.map(s => (s == null ? 0 : 1 - s.offset)); // current sensor readings used incase no data is present
        const randomizer = new Level(5, 4); // emulates a level from a neural network
        const storedTrainingData = JSON.parse(localStorage.getItem('carTrainingData'));
        const trainingExample = [];
        if(storedTrainingData){
            const inputArrays = storedTrainingData.map(entry => entry.input);
            for(let i = 0; i < inputArrays.length; i++){
                const data = Level.feedForward(inputArrays[i], randomizer); // uses a feedforward algorithem to proccess the inputs from the sensor
                trainingExample.push(data); // new procced training data
                this.trainingData.push(data); // all traindata
            }
            this.network.train(trainingExample); // trains network using only new data
            console.log("trained");
        } else{
            const result = Level.feedForward(offsets, randomizer); // creates random outputs for the sensor readings
            this.network.train([{ input: result.input, output: result.output }]);
        }
        this.trained = true;
    }
    
    
    update(roadBorders, traffic){
        if(!this.damaged){
            this.#move();
            this.polygon = this.#createPolygon();
            this.damaged = this.#assessDamage(roadBorders, traffic);
        }

        if(this.sensor){ // if there is sensor then car is 
            this.sensor.update(roadBorders,traffic);
            const offsets = this.sensor.readings.map(s =>s == null ? 0 : 1 - s.offset);
            if(this.trained === false){ // incase funtion is not trained
                this.Trainer();
            }
            const outputs = Array.from(this.network.run(offsets).map(value => (value >= 0.5 ? 1 : 0))); 

            if(this.useBrain){
                this.controls.forward = outputs[0];
                this.controls.left = outputs[1];
                this.controls.right = outputs[2];
                this.controls.reverse = outputs[3];
            }
            
            const trainingExample = { input: offsets, output: (outputs) }
            this.trainingData.push(trainingExample);
        }
    }

    #assessDamage(roadBorders,traffic){
        for(let i = 0; i < roadBorders.length; i++){
            if(polysIntersect(this.polygon,roadBorders[i])){
                return true;
            }
        }
        for(let i = 0; i < traffic.length; i++){
            if(polysIntersect(this.polygon,traffic[i].polygon)){
                return true;
            }
        }
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
        if(this.controls.forward){
            this.speed += this.acceleration;
        }
        if(this.controls.reverse){
            this.speed -= this.acceleration;
        }

        if(this.speed > this.maxSpeed){
            this.speed = this.maxSpeed;
        }
        if(this.speed <- this.maxSpeed / 2){
            this.speed = -this.maxSpeed / 2;
        }

        if(this.speed > 0){ 
            this.speed -= this.friction;
        }
        if(this.speed < 0){
            this.speed += this.friction;
        }
        if(Math.abs(this.speed) < this.friction){
            this.speed = 0;
        }

        if(this.speed != 0){
            const flip = this.speed > 0 ? 1 :- 1;
            if(this.controls.left){
                this.angle += 0.03 * flip;
            }
            if(this.controls.right){
                this.angle -= 0.03 * flip;
            }
        }

        this.x -= Math.sin(this.angle) * this.speed;
        this.y -= Math.cos(this.angle) * this.speed;
    }

    draw(ctx,color){
        if(this.damaged){
            ctx.fillStyle = "gray";
        }else{
            ctx.fillStyle = color;
        }
        ctx.beginPath();
        ctx.moveTo(this.polygon[0].x, this.polygon[0].y);
        for(let i = 1; i < this.polygon.length; i++){
            ctx.lineTo(this.polygon[i].x, this.polygon[i].y);
        }
        ctx.fill();

        if(this.sensor){
            this.sensor.draw(ctx);
        }
    }

}