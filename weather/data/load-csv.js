const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function extractColumns(data, columnNames) {
  const headers = _.first(data);

  // get indexes and pull out data
  const indexes = _.map(columnNames, column => headers.indexOf(column));
  const extracted = _.map(data, row => _.pullAt(row, indexes));

  return extracted;
}

module.exports = function loadCSV(
  filename,
  {
    dataColumns = [],
    labelColumns = [],
    converters = {},
    shuffle = false,
    splitTest = false
  }
) {
  // Read the file
  let data = fs.readFileSync(filename, { encoding: 'utf-8' });

  // split by rows, then by columns, then clean up empty data
  data = _.map(data.split('\n'), d => d.split(','));
  data = _.dropRightWhile(data, val => _.isEqual(val, ['']));

  // get headers
  const headers = _.first(data);

  // Parse each cell data
  data = _.map(data, (row, index) => {
    // Skip header row
    if (index === 0) {
      return row;
    }

    // Map each row
    return _.map(row, (cell, index) => {
      // Checks and runs function in converters option
      if (converters[headers[index]]) {
        const converted = converters[headers[index]](cell);
        return _.isNaN(converted) ? cell : converted;
      }

      // convert to a number
      const result = parseFloat(cell.replace('"', ''));
      // if result is not a number return element, else return result
      return _.isNaN(result) ? cell : result;
    });
  });

  // extract data for features and labels
  let labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);

  // removes header row
  data.shift();
  labels.shift();

  if (shuffle) {
    data = shuffleSeed.shuffle(data, 'phrase');
    labels = shuffleSeed.shuffle(labels, 'phrase');
  }

  // Split the data into a test and training set
  // can be number or true
  if (splitTest) {
    const trainSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length / 2);

    return {
      features: data.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: data.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize)
    };
  } else {
    return { features: data, labels };
  }
};
