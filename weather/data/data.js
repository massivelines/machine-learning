const parse = require('csv-parse/lib/sync');
const fs = require('fs');
const _ = require('lodash');
const { shuffle } = require('lodash');

// const LOCAL_CSV = './jena_climate_2009_2016.csv';
const LOCAL_CSV = './tester.csv';

class LoadCSVData {
  constructor(options) {
    // default options
    this.options = Object.assign(
      {
        // filename: '',
        // hasHeader: true,
        featureColumns: [],
        labelColumns: [],
        convertData: {},
        shuffle: false,
        splitTest: false,
        // configuredColumnsOnly = false,
        // fullColumnNames: string[] = null;
        // columnNamesValidated = false;
        // columnConfigs: {[key: string]: ColumnConfig} = null
        // delimiter = ',',
        // delimWhitespace = false
      },
      options
    );
  }

  extractDataSection(data, headers, columnNames) {
    // get indexes from headers
    const indexes = _.map(columnNames, (column) => headers.indexOf(column));
    // get only required data out
    const extracted = _.map(data, (row) => _.at(row, indexes));

    return extracted;
  }

  extractData(data, headers) {
    let tempData = data;

    // shuffle the data
    if (this.options.shuffle) {
      tempData = _.shuffle(tempData);
    }

    const labels = this.extractDataSection(
      tempData,
      headers,
      this.options.labelColumns
    );

    const features = this.extractDataSection(
      tempData,
      headers,
      this.options.featureColumns
    );

    // Split the data into a test and training set
    // can be number or true
    if (this.options.splitTest) {
      const trainSize = _.isNumber(this.options.splitTest)
        ? this.options.splitTest
        : Math.floor(data.length / 2);

      return {
        features: data.slice(trainSize),
        labels: labels.slice(trainSize),
        testFeatures: data.slice(0, trainSize),
        testLabels: labels.slice(0, trainSize),
      };
    }

    return { labels, features };
  }

  loadData() {
    // Read the file
    const csvContent = fs.readFileSync(`${__dirname}/${LOCAL_CSV}`, {
      encoding: 'utf-8',
    });

    // Split into main rows at new lines, then split at ',', then clean up any empty cells at the end of the row
    const cleanedData = _.map(csvContent.split('\n'), (mainRow) =>
      _.chain(mainRow)
        .split(',')
        .dropRightWhile((val) => _.isEqual(val, ''))
        .value()
    );

    // get headers and remove "
    const headers = _.head(cleanedData).map((item) =>
      _.replace(item, /"/g, '')
    );

    // Remove header row then map each row and cell, if has converters call function, else try to parse parseFloat
    const formattedData = _.chain(cleanedData)
      .slice(1)
      .map((row) =>
        _.map(row, (cellData, index) => {
          const currentColumn = headers[index];

          // If convertData has items inside it and has current column
          if (
            Object.keys(this.options.convertData).length > 0 &&
            !!this.options.convertData[currentColumn]
          ) {
            if (typeof this.options.convertData[currentColumn] === 'function') {
              // if convertData item is a function call and return value
              const converted = this.options.convertData[currentColumn](
                cellData
              );
              return _.isNaN(converted) ? cellData : converted;
            }
            // else it is a regex and returns true or false
            const match = new RegExp(this.options.convertData[currentColumn]);
            return match.test(cellData);
          }

          // Else parseFloat the cell and replace un-needed string items
          // if result is not a value return original value, else return result
          const result = parseFloat(cellData.replace('"', ''));
          return _.isNaN(result) ? cellData : result;
        })
      )
      .value();

    // Extract data for features and labels
    return this.extractData(formattedData, headers);
  }
}

module.exports = LoadCSVData;

// const loader = new LoadCSVData({
//   featureColumns: ['test value', 'id'],
//   labelColumns: ['passed'],
//   convertData: {
//     passed: (data) => (data === 'True' ? true : false),
//   },
//   shuffle: true,
//   // splitTest: -1,
// });
// const tester = loader.loadData();
// console.log(tester);
