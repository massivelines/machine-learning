/**
 * Notes
 * Mean square error
 * b = y intercept on a xy graph (0, b), approximate line that runs through all points
 * where it hits the y axis
 * m = slope of the line
 */

const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];
    // this.bHistory = [];

    // default options
    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
      },
      options
    );

    // (m, b)
    // generate tensor with zeros (dynamic columns)
    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent(features, labels) {
    // Mean square error (MSE)
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    // (bSlope, mSlope)
    // (transpose, have to change direction of array so that it can be multiplied)
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    // (m, b)
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    // number of batches to run gradientDescent with
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    // for (let i = 0; i < this.options.iterations; i++) {
    //   // console.log(this.options.learningRate)
    //   // this.bHistory.push(this.weights.get(0, 0));
    //   this.gradientDescent();
    //   this.recordMSE();
    //   this.updateLearningRate();
    // }

    // setup for batch
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );

        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    // Sum of Squares of Residuals
    const res = testLabels.sub(predictions).pow(2).sum().get();

    // Total Sum of Squares
    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    // Coefficient of Determination
    // (return - number if Residuals is larger than total, need more work to increase accuracy/analyses )
    return 1 - res / tot;
  }

  processFeatures(features) {
    // (if use mean and variance on features, must also use on testFeatures)
    features = tf.tensor(features);

    // if mean and variance are defined, calculate, else do first time standardize()
    // (could be moved to standardize)
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    // generate column of 1,s, merge with features
    // (must be after standardize, else effect column of 1's, (javaScript rounding errors))
    // (this.features.shape[0] is length of data)
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    // undefined on build,
    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  // Records MSE to array, newest first
  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    this.mseHistory.unshift(mse);
  }

  // Self adjusting leaning rate
  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    // if mseHistory is increasing (more incorrect), reduce leaning rate
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;

// gradientDescent() {
//   const currentGuessesForMPG = this.features.map((row) => {
//     return this.m * row[0] + this.b;
//   });
//   const bSlope =
//     (_.sum(
//       currentGuessesForMPG.map((guess, i) => {
//         return guess - this.labels[i][0];
//       })
//     ) *
//       2) /
//     this.features.length;
//   const mSlope =
//     (_.sum(
//       currentGuessesForMPG.map((guess, i) => {
//         return -1 * this.features[i][0] * (this.labels[i][0] - guess);
//       })
//     ) *
//       2) /
//     this.features.length;
//     this.m = this.m - mSlope * this.options.learningRate;
//     this.b = this.b - bSlope * this.options.learningRate;
// }
