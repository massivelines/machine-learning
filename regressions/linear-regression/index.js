require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

regression.train();
// (if infinity returned adjust learningRate)
// (if - number better off using average of value, but need more work to increase accuracy/analyses, adjust learningRate and iterations )

// Used only for development to check feedback
const r2 = regression.test(testFeatures, testLabels);

plot({
  // x: regression.bHistory,
  // y: regression.mseHistory.reverse(),
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  // xLabel: 'Value of B',
  yLabel: 'Mean Squared Error',
});

console.log('R2 is : ', r2);

regression
  .predict([
    [120, 2, 380],
    // [135, 2.1, 420],
  ])
  .print();
