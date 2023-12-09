const carCanvas = document.getElementById("carCanvas");
carCanvas.width = 200;

const carCtx = carCanvas.getContext("2d");

const road = new Road(carCanvas.width / 2, carCanvas.width * 0.9);

const traffic = [ // 300-400 pixles seems to be ideal distance for the "dummy" cars
    new Car(road.getLaneCenter(1), -100, 30, 50, "DUMMY", 1), // Car(lane, distance from start, width, lenght, controlType, speed)
    new Car(road.getLaneCenter(0), -400, 30, 50, "DUMMY", 1),
    new Car(road.getLaneCenter(2), -700, 30, 50, "DUMMY", 1), 
    new Car(road.getLaneCenter(0), -700, 30, 50, "DUMMY", 1),
    new Car(road.getLaneCenter(1), -1000, 30, 50, "DUMMY", 1), 
    new Car(road.getLaneCenter(0), -1400, 30, 50, "DUMMY", 1),
    new Car(road.getLaneCenter(0), -1700, 30, 50, "DUMMY", 1), 
    new Car(road.getLaneCenter(2), -1700, 30, 50, "DUMMY", 1),
    new Car(road.getLaneCenter(1), -2000, 30, 50, "DUMMY", 1), 
    new Car(road.getLaneCenter(0), -2400, 30, 50, "DUMMY", 1),
    new Car(road.getLaneCenter(2), -2700, 30, 50, "DUMMY", 1), 
    new Car(road.getLaneCenter(0), -3000, 30, 50, "DUMMY", 1),
];

async function generateCars(N) {
    const cars = [];
    const storedTrainingData = JSON.parse(localStorage.getItem('carTrainingData'));
    for (let i = 0; i < N; i++) {
        try {
            if (i == 1 && storedTrainingData) { // first car get the og brain and keeps it old behavior
                const car = new Car(road.getLaneCenter(1), 100, 30, 50, "AI", 4);
                try {
                    car.network = await tf.loadLayersModel('localstorage://bestModel');
                } catch (error) {
                    console.log("Error loading network: ", error);
                }
                cars.push(car);
            } else if (storedTrainingData && i <= N / 2) { // half the cars get a mutated brain and train using the new data
                const car = new Car(road.getLaneCenter(1), 100, 30, 50, "AI", 4);
                try {
                    const model = await tf.loadLayersModel('localstorage://bestModel');
                    car.network = await mutateModel(model, 0.1);
                } catch (error) {
                    console.log("Error loading network: ", error);
                }
                try {
                    if (car.trained == false) 
                        await car.Trainer();
                } catch (error) {
                    console.log("Error while training networks: ", error);
                }
                cars.push(car);
            } else { // the rest of the cars get a fresh brain with and train using the new data
                const car = new Car(road.getLaneCenter(1), 100, 30, 50, "AI", 4);
                try {
                    if (car.trained == false) 
                        await car.Trainer();
                } catch (error) {
                    console.log("Error while training networks: ", error);
                }
                cars.push(car);
            }
        } catch (error) {
            console.error("Error creating car:", error);
        }
    }
    return cars;
}

async function main() {
    const cars = await generateCars(50);
    animate(cars);
}
let bestCar = [];

main();


function animate(data) {
    let cars = data;
    bestCar = cars[0];
    try {
        for (let i = 0; i < traffic.length; i++) 
            traffic[i].update(road.borders, []);
    } catch (error) {
        console.error("Error updating traffic", error);
    }
    try {
        for (let i = 0; i < cars.length; i++) 
            cars[i].update(road.borders, traffic);
    } catch (error) {
        console.error("Error updating traffic", error);
    }
    try {
        for (let i = 0; i < cars.length; i++)
            if (cars[i].y !== undefined && (bestCar === undefined || cars[i].y < bestCar.y)) 
                bestCar = cars[i];
    } catch (error) {
        console.error("Error updating traffic", error);
    }
    try {
        for (let i = 0; i < traffic.length; i++) 
            traffic[i].update(road.borders, []);   
    } catch (error) {
        console.error("Error updating traffic", error);
    }

    carCanvas.height = window.innerHeight;
    carCtx.save();
    carCtx.translate(0, -bestCar.y + carCanvas.height * 0.7);

    road.draw(carCtx);

    for (let i = 0; i < traffic.length; i++) 
        traffic[i].draw(carCtx, "red");

    carCtx.globalAlpha = 0.2;

    for (let i = 0; i < cars.length; i++) 
        cars[i].draw(carCtx, "blue");

    carCtx.globalAlpha = 1;
    bestCar.draw(carCtx, "blue", true);

    carCtx.restore();
    requestAnimationFrame(() => animate(data));
}


