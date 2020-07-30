/**
 * Multiple Classification Options
 * Copy of logistic-regression.js
 * Goal: predict if a car has: low, medium, or high full efficiency
 * Labels: mpg
 * Features: horsepower, displacement, weight
 * Notes:
 * Marginal Probability Distribution
 * - different probabilities are not connected
 * - (can be in only one classification at a time [1,0,0] )
 * Conditional Probability Distribution
 * - different probabilities are interconnected
 * - (can be in different classifications at the same time [1,1,0] )
 * - add up all probabilities if >1 usually means conditional
 */

require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const MultiLogisticRegression = require('./multi-logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    // UPDATED:
    // labelColumns: ['passedemissions'],
    labelColumns: ['mpg'],
    converters: {
      // passedemissions: (value) => (value === 'TRUE' ? 1 : 0),
      mpg: (value) => {
        const mpg = parseFloat(value);

        if (mpg < 15) {
          return [1, 0, 0];
        } else if (mpg < 30) {
          return [0, 1, 0];
        } else {
          return [0, 0, 1];
        }
      },
    },
  }
);

//UPDATED: CSV > Converter doesn't handle arrays very well [[[],[]]], it nests them so just flatten once for [[],[]]
const regression = new MultiLogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.6,
});

// UPDATED: just to check if weights is in correct matrix format
// regression.weights.print()

regression.train();

// regression.predict([[215, 440, 2.16]]).print();

// Causes [[1, 1, 0]]
// regression.predict([[150, 200, 2.28]]).print();

const test = regression.test(testFeatures, _.flatMap(testLabels));
console.log(test);

// plot({
//   x: regression.costHistory.reverse(),
//   xLabel: 'Iteration #',
//   yLabel: 'Cost',
// });

// console.log(regression.test(testFeatures, testLabels));
