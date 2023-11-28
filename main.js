const carCanvas=document.getElementById("carCanvas");
carCanvas.width = 200;

const carCtx = carCanvas.getContext("2d");

const road = new Road(carCanvas.width/2,carCanvas.width*0.9);

const N = 50; //number of AI cars
const cars = generateCars(N);
let bestCar = cars[0]; // smartest car ie the one who makes it furtherest

const traffic=[ // 300 pixles seems to be ideal distance for the "dummy" cars
    new Car(road.getLaneCenter(1), -100, 30, 50, "DUMMY", 1), // Car(lane, distance from start, width, lenght, controlType, speed)
    new Car(road.getLaneCenter(0), -400, 30, 50, "DUMMY", 1),
    new Car(road.getLaneCenter(2), -700, 30, 50, "DUMMY", 1), // Car(lane, distance from start, width, lenght, controlType, speed)
    new Car(road.getLaneCenter(0), -700, 30, 50, "DUMMY", 1),
    new Car(road.getLaneCenter(1), -1000, 30, 50, "DUMMY", 1), // Car(lane, distance from start, width, lenght, controlType, speed)
    new Car(road.getLaneCenter(0), -1400, 30, 50, "DUMMY", 1),
];

function save(){ // saves the smartest car to localStorage
    try {
        if (bestCar && bestCar.network) {
            const json = bestCar.network.toJSON();
            localStorage.setItem("bestBrain", JSON.stringify(json));
            localStorage.setItem('carTrainingData', JSON.stringify(bestCar.trainingData));
            console.log("Save successful");
            console.log(bestCar.trainingData);
        } else {
            console.error("No best car or network found to save.");
        }
    } catch (error) {
        console.error("Error saving brain and training data:", error);
    }
}

function discard(){
    try{
        localStorage.removeItem("bestBrain");
        localStorage.removeItem("carTrainingData");
        console.log("delete succesful");
    } catch (error) {
        console.error("Error deleting brain and training data:", error);
    }
}

function generateCars(N) {
    const cars = [];
    for (let i = 1; i <= N; i++) {
        if (i === 1) {
            const storedBrain = localStorage.getItem("bestBrain");
            try {
                const brainData = storedBrain ? JSON.parse(storedBrain) : null;
                cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI", 4, brainData));
            } catch (error) {
                console.error("Error parsing stored brain data:", error);
                cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI", 4));
            }
        } else {
            const storedBrain = localStorage.getItem("bestBrain");
            try {
                const brainData = storedBrain ? JSON.parse(storedBrain) : null;
                cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI", 4, brainData, true));
            } catch(error) {
                cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI", 4, null, true));
                console.error("Error parsing stored brain data:", error);
            }
        }
    }
    return cars;
}


animate();

function animate(){
    try{
        for(let i=0;i<traffic.length;i++) { //update the location of "dummy" traffic cars
            traffic[i].update(road.borders,[]);
        }
        for(let i=0;i<cars.length;i++) {
            cars[i].update(road.borders, traffic);
        }
        bestCar = cars.find(c =>c.y  == Math.min(...cars.map(c =>c.y)));
        for (let i = 0; i < traffic.length; i++) {
            traffic[i].update(road.borders, []);
        }
    } catch (error) {
        console.error("Error updating traffic", error);
    }
    carCanvas.height = window.innerHeight; 

    carCtx.save();
    carCtx.translate(0, -bestCar.y + carCanvas.height * 0.7); // centers car away from the top of the screen

    road.draw(carCtx);
    
    for(let i = 0; i < traffic.length; i++) {
        traffic[i].draw(carCtx, "red");
    }

    carCtx.globalAlpha = 0.2; // makes all the other AI cars transpernt

    for(let i = 0; i < cars.length; i++) {
        cars[i].draw(carCtx, "blue");
    }

    carCtx.globalAlpha = 1;
    bestCar.draw(carCtx, "blue", true);

    carCtx.restore();
    requestAnimationFrame(animate);
}