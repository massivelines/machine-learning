/**
 * Multiple Classification Options
 * Goal: predict number
 * Labels: mpg
 * Features: horsepower, displacement, weight
 */

//  node --max-old-space-size=4096 index.js

// require('@tensorflow/tfjs-node');
// require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs-node-gpu');
// const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./mnist-image-recognition');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

// UPDATED:
// move everything below inside of function so it will be garbage collected
// const mnistData = mnist.training(0, 20000);

// const features = mnistData.images.values.map((image) => _.flatMap(image));
// const encodedLabels = mnistData.labels.values.map((label) => {
//   const row = new Array(10).fill(0);
//   row[label] = 1;
//   return row;
// });

function loadData() {
  const mnistData = mnist.training(0, 60000);

  const features = mnistData.images.values.map((image) => _.flatMap(image));
  const encodedLabels = mnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { features, labels: encodedLabels };
}

const { features, labels } = loadData();

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 500,
});

regression.train();

// stop here for memory profiling after train
// debugger;

const testMnistData = mnist.testing(0, 10000);
const testFeatures = testMnistData.images.values.map((image) =>
  _.flatMap(image)
);
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Accuracy is', accuracy);

plot({
  x: regression.costHistory.reverse()
})
// console.log(regression.costHistory)